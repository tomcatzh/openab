use crate::acp::connection::AcpConnection;
use crate::acp::protocol::ConfigOption;
use crate::config::{AgentConfig, WorkspaceConfig, WorkspaceRequest};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio::time::Instant;
use tracing::{info, warn};

/// Combined state protected by a single lock to prevent deadlocks.
/// Lock ordering: never await a per-connection mutex while holding `state`.
struct PoolState {
    /// Active connections: thread_key → AcpConnection handle.
    active: HashMap<String, Arc<Mutex<AcpConnection>>>,
    /// Lock-free cancel handles: thread_key → (stdin, session_id).
    /// Stored separately so cancel can work without locking the connection.
    cancel_handles: HashMap<String, (Arc<tokio::sync::Mutex<tokio::process::ChildStdin>>, String)>,
    /// Suspended sessions: thread_key → ACP sessionId.
    /// Used at runtime to decide which thread can be resumed via `session/load`
    /// because it no longer has a live in-memory connection.
    suspended: HashMap<String, String>,
    /// Persisted resumable sessions: thread_key → ACP sessionId.
    /// Includes both suspended sessions and active sessions so a process restart
    /// can recover any live thread via `session/load`.
    persisted: HashMap<String, String>,
    /// Persisted project working directories: thread_key → cwd.
    /// This is intentionally separate from `persisted` so rolling back to
    /// stock OpenAB leaves the original thread/session map untouched.
    workdirs: HashMap<String, String>,
    /// Serializes create/resume work per thread so rapid same-thread requests
    /// cannot race each other into duplicate `session/load` attempts.
    creating: HashMap<String, Arc<Mutex<()>>>,
}

pub struct SessionPool {
    state: RwLock<PoolState>,
    config: AgentConfig,
    max_sessions: usize,
    mapping_path: PathBuf,
    workdir_mapping_path: PathBuf,
    per_thread_workdir: bool,
    workspace: WorkspaceConfig,
}

type EvictionCandidate = (String, Arc<Mutex<AcpConnection>>, Instant, Option<String>);

fn remove_if_same_handle<T>(
    map: &mut HashMap<String, Arc<Mutex<T>>>,
    key: &str,
    expected: &Arc<Mutex<T>>,
) -> Option<Arc<Mutex<T>>> {
    let should_remove = map
        .get(key)
        .is_some_and(|current| Arc::ptr_eq(current, expected));
    if should_remove {
        map.remove(key)
    } else {
        None
    }
}

fn get_or_insert_gate(map: &mut HashMap<String, Arc<Mutex<()>>>, key: &str) -> Arc<Mutex<()>> {
    map.entry(key.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

impl SessionPool {
    pub fn new(
        config: AgentConfig,
        max_sessions: usize,
        per_thread_workdir: bool,
        workspace: WorkspaceConfig,
    ) -> Self {
        let openab_dir = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp"))
            .join(".openab");
        let _ = std::fs::create_dir_all(&openab_dir);
        let mapping_path = openab_dir.join("thread_map.json");
        let workdir_mapping_path = openab_dir.join("thread_workdir_map.json");
        let suspended = Self::load_mapping(&mapping_path);
        let workdirs = Self::load_mapping(&workdir_mapping_path);
        Self {
            state: RwLock::new(PoolState {
                active: HashMap::new(),
                cancel_handles: HashMap::new(),
                persisted: suspended.clone(),
                suspended,
                workdirs,
                creating: HashMap::new(),
            }),
            config,
            max_sessions,
            mapping_path,
            workdir_mapping_path,
            per_thread_workdir,
            workspace,
        }
    }

    pub fn workspace_enabled(&self) -> bool {
        self.workspace.root.is_some()
    }

    fn load_mapping(path: &Path) -> HashMap<String, String> {
        match std::fs::read_to_string(path) {
            Ok(data) => serde_json::from_str(&data).unwrap_or_else(|e| {
                warn!(path = %path.display(), error = %e, "corrupt thread_map.json, starting fresh");
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        }
    }

    fn save_mapping(&self, persisted: &HashMap<String, String>) {
        Self::save_mapping_at(&self.mapping_path, persisted);
    }

    fn save_workdir_mapping(&self, workdirs: &HashMap<String, String>) {
        Self::save_mapping_at(&self.workdir_mapping_path, workdirs);
    }

    fn save_mapping_at(path: &Path, persisted: &HashMap<String, String>) {
        let data = match serde_json::to_string_pretty(persisted) {
            Ok(d) => d,
            Err(e) => {
                warn!(error = %e, "failed to serialize thread mapping");
                return;
            }
        };
        let tmp = path.with_extension("json.tmp");
        if let Err(e) = std::fs::write(&tmp, &data).and_then(|_| std::fs::rename(&tmp, path)) {
            warn!(path = %path.display(), error = %e, "failed to persist thread mapping");
        }
    }

    fn safe_thread_component(thread_id: &str) -> Result<String> {
        let mut out = String::with_capacity(thread_id.len());
        for b in thread_id.bytes() {
            match b {
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' => out.push(b as char),
                _ => out.push_str(&format!("_{b:02x}")),
            }
        }
        if out.is_empty() || out == "." || out == ".." {
            return Err(anyhow!("invalid thread_id: {thread_id}"));
        }
        Ok(out)
    }

    fn has_unsafe_components(path: &Path) -> bool {
        path.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::CurDir | Component::Prefix(_)
            )
        })
    }

    fn workspace_root(&self) -> Option<PathBuf> {
        self.workspace.root.as_ref().map(PathBuf::from)
    }

    fn validate_workspace_name(name: &str) -> Result<PathBuf> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(anyhow!("empty workspace directive"));
        }
        let rel = PathBuf::from(trimmed);
        if rel.is_absolute() {
            return Err(anyhow!(
                "workspace must be a relative path under the workspace root"
            ));
        }
        if Self::has_unsafe_components(&rel) {
            return Err(anyhow!("workspace contains unsafe path components"));
        }
        if rel.as_os_str().is_empty() || trimmed == "." || trimmed == ".." {
            return Err(anyhow!("invalid workspace path"));
        }
        Ok(rel)
    }

    fn validate_existing_workspace_path(&self, requested: &Path, root: &Path) -> Result<String> {
        if !requested.is_dir() {
            return Err(anyhow!(
                "workspace is not a directory: {}",
                requested.display()
            ));
        }
        let root = std::fs::canonicalize(root)
            .map_err(|e| anyhow!("failed to resolve workspace root {}: {e}", root.display()))?;
        let canonical = std::fs::canonicalize(requested)
            .map_err(|e| anyhow!("failed to resolve workspace {}: {e}", requested.display()))?;
        if canonical.starts_with(&root) {
            return Ok(canonical.to_string_lossy().to_string());
        }
        Err(anyhow!("workspace is outside the workspace root"))
    }

    fn existing_ancestor(path: &Path) -> Option<PathBuf> {
        let mut current = path.parent();
        while let Some(candidate) = current {
            if candidate.exists() {
                return Some(candidate.to_path_buf());
            }
            current = candidate.parent();
        }
        None
    }

    fn validate_requested_workspace(&self, request: &WorkspaceRequest) -> Result<String> {
        let Some(root) = self.workspace_root() else {
            return Ok(self.config.working_dir.clone());
        };
        let rel = Self::validate_workspace_name(request.name())?;
        let requested = root.join(&rel);
        if !requested.starts_with(&root) {
            return Err(anyhow!("workspace is outside the workspace root"));
        }
        if requested.exists() {
            if request.creates_missing() {
                return Err(anyhow!(
                    "workspace already exists: {}; use [[ws:{}]] for an existing workspace",
                    requested.display(),
                    request.name()
                ));
            }
            return self.validate_existing_workspace_path(&requested, &root);
        }

        if !request.creates_missing() {
            return Err(anyhow!(
                "workspace does not exist: {}; use [[ws:{} --create]] to create it",
                requested.display(),
                request.name()
            ));
        }

        let root_canonical = std::fs::canonicalize(&root)
            .map_err(|e| anyhow!("failed to resolve workspace root {}: {e}", root.display()))?;
        let ancestor = Self::existing_ancestor(&requested)
            .ok_or_else(|| anyhow!("workspace root does not exist: {}", root.display()))?;
        let ancestor_canonical = std::fs::canonicalize(&ancestor).map_err(|e| {
            anyhow!(
                "failed to resolve workspace parent {}: {e}",
                ancestor.display()
            )
        })?;
        if !ancestor_canonical.starts_with(&root_canonical) {
            return Err(anyhow!("workspace parent is outside the workspace root"));
        }

        std::fs::create_dir_all(&requested)
            .map_err(|e| anyhow!("failed to create workspace {}: {e}", requested.display()))?;
        self.validate_existing_workspace_path(&requested, &root)
    }

    pub fn prepare_workspace_request(
        &self,
        workspace_request: Option<&WorkspaceRequest>,
    ) -> Result<Option<WorkspaceRequest>> {
        if !self.workspace_enabled() {
            return Ok(None);
        }

        let Some(requested) = workspace_request else {
            if self.workspace.required {
                return Err(anyhow!(
                    "missing workspace directive; start the thread with [[ws:<name>]] for an existing workspace or [[ws:<name> --create]] to create one"
                ));
            }
            return Ok(None);
        };

        let rel = Self::validate_workspace_name(requested.name())?;
        let workspace_name = rel.to_string_lossy().to_string();
        self.validate_requested_workspace(requested)?;
        Ok(Some(WorkspaceRequest::Existing(workspace_name)))
    }

    async fn resolve_working_dir(
        &self,
        thread_id: &str,
        workspace_request: Option<&WorkspaceRequest>,
    ) -> Result<String> {
        if self.workspace_enabled() {
            let existing_workdir = {
                let state = self.state.read().await;
                state.workdirs.get(thread_id).cloned()
            };

            if let Some(existing) = existing_workdir {
                if workspace_request.is_some() {
                    return Err(anyhow!(
                        "thread already has workspace {existing}; start a new thread to change workspace"
                    ));
                }
                return Ok(existing);
            }

            if let Some(requested) = workspace_request {
                let cwd = self.validate_requested_workspace(requested)?;
                let mut state = self.state.write().await;
                if let Some(existing) = state.workdirs.get(thread_id) {
                    return Err(anyhow!(
                        "thread already has workspace {existing}; start a new thread to change workspace"
                    ));
                }
                state.workdirs.insert(thread_id.to_string(), cwd.clone());
                self.save_workdir_mapping(&state.workdirs);
                info!(thread_id, working_dir = %cwd, "bound thread workspace");
                return Ok(cwd);
            }

            if self.workspace.required {
                return Err(anyhow!(
                    "missing workspace directive; start the thread with [[ws:<name>]] for an existing workspace or [[ws:<name> --create]] to create one"
                ));
            }
        }

        if self.per_thread_workdir {
            let safe = Self::safe_thread_component(thread_id)?;
            let dir = PathBuf::from(&self.config.working_dir)
                .join("sessions")
                .join(safe);
            tokio::fs::create_dir_all(&dir).await?;
            return Ok(dir.to_string_lossy().to_string());
        }

        Ok(self.config.working_dir.clone())
    }

    pub async fn get_or_create(
        &self,
        thread_id: &str,
        workspace_request: Option<&WorkspaceRequest>,
    ) -> Result<()> {
        let create_gate = {
            let mut state = self.state.write().await;
            get_or_insert_gate(&mut state.creating, thread_id)
        };
        let _create_guard = create_gate.lock().await;
        let working_dir = self
            .resolve_working_dir(thread_id, workspace_request)
            .await?;

        let (existing, saved_session_id) = {
            let state = self.state.read().await;
            (
                state.active.get(thread_id).cloned(),
                state.suspended.get(thread_id).cloned(),
            )
        };

        let had_existing = existing.is_some();
        let mut saved_session_id = saved_session_id;
        if let Some(conn) = existing.clone() {
            let conn = conn.lock().await;
            if conn.alive() {
                return Ok(());
            }
            if saved_session_id.is_none() {
                saved_session_id = conn.acp_session_id.clone();
            }
        }

        // Snapshot active handles so we can inspect them outside the state lock.
        let snapshot: Vec<(String, Arc<Mutex<AcpConnection>>)> = {
            let state = self.state.read().await;
            state
                .active
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
                .collect()
        };

        let mut eviction_candidate: Option<EvictionCandidate> = None;
        let mut skipped_locked_candidates = 0usize;
        for (key, conn) in snapshot {
            if key == thread_id {
                continue;
            }
            let conn_handle = Arc::clone(&conn);
            let Ok(conn) = conn.try_lock() else {
                skipped_locked_candidates += 1;
                continue;
            };
            let candidate = (
                key,
                conn_handle,
                conn.last_active,
                conn.acp_session_id.clone(),
            );
            match &eviction_candidate {
                Some((_, _, oldest_last_active, _)) if candidate.2 >= *oldest_last_active => {}
                _ => eviction_candidate = Some(candidate),
            }
        }

        // Build the replacement connection outside the state lock so one stuck
        // initialization does not block all unrelated sessions.
        let mut new_conn = AcpConnection::spawn(
            &self.config.command,
            &self.config.args,
            &working_dir,
            &self.config.env,
            &self.config.inherit_env,
        )
        .await?;

        new_conn.initialize().await?;

        let mut resumed = false;
        if let Some(ref sid) = saved_session_id {
            if new_conn.supports_load_session {
                match new_conn.session_load(sid, &working_dir).await {
                    Ok(()) => {
                        info!(thread_id, session_id = %sid, "session resumed via session/load");
                        resumed = true;
                    }
                    Err(e) => {
                        warn!(thread_id, session_id = %sid, error = %e, "session/load failed, creating new session");
                    }
                }
            }
        }

        if !resumed {
            new_conn.session_new(&working_dir).await?;
            // Surface the reset banner both for restored sessions and for stale
            // live entries that died before we could recover a resumable
            // session id. In both cases the caller is continuing after an
            // unexpected session loss.
            if had_existing || saved_session_id.is_some() {
                new_conn.session_reset = true;
            }
        }

        let cancel_handle = new_conn.cancel_handle();
        let cancel_session_id = new_conn.acp_session_id.clone().unwrap_or_default();
        let new_conn = Arc::new(Mutex::new(new_conn));

        let mut state = self.state.write().await;

        // Another task may have created a healthy connection while we were
        // initializing this one.
        if let Some(existing) = state.active.get(thread_id).cloned() {
            let Ok(existing) = existing.try_lock() else {
                return Ok(());
            };
            if existing.alive() {
                return Ok(());
            }
            warn!(thread_id, "stale connection, rebuilding");
            drop(existing);
            state.active.remove(thread_id);
            state.cancel_handles.remove(thread_id);
        }

        if state.active.len() >= self.max_sessions {
            if let Some((key, expected_conn, _, sid)) = eviction_candidate {
                if remove_if_same_handle(&mut state.active, &key, &expected_conn).is_some() {
                    state.cancel_handles.remove(&key);
                    info!(evicted = %key, "pool full, suspending oldest idle session");
                    if let Some(sid) = sid {
                        state.persisted.insert(key.clone(), sid.clone());
                        state.suspended.insert(key, sid);
                    } else {
                        state.persisted.remove(&key);
                    }
                } else {
                    warn!(evicted = %key, "pool full but eviction candidate changed before removal");
                }
            } else if skipped_locked_candidates > 0 {
                warn!(
                    max_sessions = self.max_sessions,
                    skipped_locked_candidates,
                    "pool full but all other sessions were busy during eviction scan"
                );
            }
        }

        if state.active.len() >= self.max_sessions {
            return Err(anyhow!("pool exhausted ({} sessions)", self.max_sessions));
        }

        if cancel_session_id.is_empty() {
            state.persisted.remove(thread_id);
        } else {
            state
                .persisted
                .insert(thread_id.to_string(), cancel_session_id.clone());
        }
        state.suspended.remove(thread_id);
        state.active.insert(thread_id.to_string(), new_conn);
        if !cancel_session_id.is_empty() {
            state
                .cancel_handles
                .insert(thread_id.to_string(), (cancel_handle, cancel_session_id));
        }
        self.save_mapping(&state.persisted);
        Ok(())
    }

    /// Get mutable access to a connection. Caller must have called get_or_create first.
    ///
    /// Only the per-connection `Mutex` is held during `f`; the pool-level
    /// `RwLock` is acquired briefly (read-only) to look up the `Arc` and then
    /// released, so other connections can be used concurrently.
    pub async fn with_connection<F, R>(&self, thread_id: &str, f: F) -> Result<R>
    where
        F: for<'a> FnOnce(
            &'a mut AcpConnection,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<R>> + Send + 'a>,
        >,
    {
        let conn = {
            let state = self.state.read().await;
            state
                .active
                .get(thread_id)
                .cloned()
                .ok_or_else(|| anyhow!("no connection for thread {thread_id}"))?
        };

        let mut conn = conn.lock().await;
        f(&mut conn).await
    }

    /// Get cached configOptions for a session (e.g. available models).
    pub async fn get_config_options(&self, thread_id: &str) -> Vec<ConfigOption> {
        let state = self.state.read().await;
        let conn = match state.active.get(thread_id) {
            Some(c) => c.clone(),
            None => return Vec::new(),
        };
        drop(state);
        let conn = conn.lock().await;
        conn.config_options.clone()
    }

    /// Set a config option (e.g. model) via ACP and return updated options.
    pub async fn set_config_option(
        &self,
        thread_id: &str,
        config_id: &str,
        value: &str,
    ) -> Result<Vec<ConfigOption>> {
        let conn = {
            let state = self.state.read().await;
            state
                .active
                .get(thread_id)
                .cloned()
                .ok_or_else(|| anyhow!("no connection for thread {thread_id}"))?
        };
        let mut conn = conn.lock().await;
        conn.set_config_option(config_id, value).await
    }

    /// Cancel the current in-flight operation for a session.
    /// Uses pre-stored cancel handles to avoid locking the connection (which is held during streaming).
    pub async fn cancel_session(&self, thread_id: &str) -> Result<()> {
        let (stdin, session_id) = {
            let state = self.state.read().await;
            state
                .cancel_handles
                .get(thread_id)
                .cloned()
                .ok_or_else(|| anyhow!("no session for thread {thread_id}"))?
        };
        let data = serde_json::to_string(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": "session/cancel",
            "params": {"sessionId": session_id}
        }))?;
        tracing::info!(session_id, "sending session/cancel");
        use tokio::io::AsyncWriteExt;
        let mut w = stdin.lock().await;
        w.write_all(data.as_bytes()).await?;
        w.write_all(b"\n").await?;
        w.flush().await?;
        Ok(())
    }

    /// Reset a session: cancel any in-flight operation, remove the active connection,
    /// and clear all suspended state. The ACP process will be killed once the last
    /// Arc reference is dropped (after streaming finishes). The next message will
    /// trigger a fresh `get_or_create` with a new ACP session.
    pub async fn reset_session(&self, thread_id: &str) -> Result<()> {
        // Send session/cancel via the lock-free stdin handle first.
        // This stops in-flight streaming even while with_connection() holds the
        // connection mutex, so the old process finishes promptly.
        if let Some((stdin, session_id)) = {
            let state = self.state.read().await;
            state.cancel_handles.get(thread_id).cloned()
        } {
            let data = serde_json::to_string(&serde_json::json!({
                "jsonrpc": "2.0",
                "method": "session/cancel",
                "params": {"sessionId": session_id}
            }))?;
            tracing::info!(session_id, "reset: sending session/cancel");
            use tokio::io::AsyncWriteExt;
            let mut w = stdin.lock().await;
            let _ = w.write_all(data.as_bytes()).await;
            let _ = w.write_all(b"\n").await;
            let _ = w.flush().await;
        }

        let mut state = self.state.write().await;
        let had_active = state.active.remove(thread_id).is_some();
        state.cancel_handles.remove(thread_id);
        state.suspended.remove(thread_id);
        state.persisted.remove(thread_id);
        state.creating.remove(thread_id);
        self.save_mapping(&state.persisted);
        if had_active {
            info!(thread_id, "session reset");
            Ok(())
        } else {
            Err(anyhow!("no session for thread {thread_id}"))
        }
    }

    pub async fn cleanup_idle(&self, ttl_secs: u64) {
        let cutoff = Instant::now() - std::time::Duration::from_secs(ttl_secs);

        let snapshot: Vec<(String, Arc<Mutex<AcpConnection>>)> = {
            let state = self.state.read().await;
            state
                .active
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
                .collect()
        };

        let mut stale = Vec::new();
        for (key, conn) in snapshot {
            // Skip active sessions for this cleanup round instead of waiting on
            // their per-connection mutex. A busy session is not idle.
            let conn_handle = Arc::clone(&conn);
            let Ok(conn) = conn.try_lock() else {
                continue;
            };
            if conn.last_active < cutoff || !conn.alive() {
                stale.push((key, conn_handle, conn.acp_session_id.clone()));
            }
        }

        if stale.is_empty() {
            return;
        }

        let mut state = self.state.write().await;
        for (key, expected_conn, sid) in stale {
            if remove_if_same_handle(&mut state.active, &key, &expected_conn).is_some() {
                info!(thread_id = %key, "cleaning up idle session");
                state.cancel_handles.remove(&key);
                if let Some(sid) = sid {
                    state.persisted.insert(key.clone(), sid.clone());
                    state.suspended.insert(key, sid);
                } else {
                    state.persisted.remove(&key);
                }
            }
        }
        self.save_mapping(&state.persisted);
    }

    pub async fn shutdown(&self) {
        // Snapshot active handles, then drop state lock before awaiting
        // per-connection mutexes (lock ordering: never hold state while
        // awaiting a connection lock).
        let snapshot: Vec<(String, Arc<Mutex<AcpConnection>>)> = {
            let state = self.state.read().await;
            state
                .active
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
                .collect()
        };

        let mut session_ids: Vec<(String, String)> = Vec::new();
        for (key, conn) in snapshot {
            let conn = conn.lock().await;
            if let Some(sid) = conn.acp_session_id.clone() {
                session_ids.push((key, sid));
            }
        }

        let mut state = self.state.write().await;
        for (key, sid) in session_ids {
            state.persisted.insert(key.clone(), sid.clone());
            state.suspended.insert(key, sid);
        }
        self.save_mapping(&state.persisted);
        let count = state.active.len();
        state.active.clear();
        state.cancel_handles.clear();
        info!(count, "pool shutdown complete");
    }
}

#[cfg(test)]
mod tests {
    use super::{get_or_insert_gate, remove_if_same_handle, SessionPool};
    use crate::config::{AgentConfig, WorkspaceConfig, WorkspaceRequest};
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Mutex;

    fn agent_config(working_dir: &Path) -> AgentConfig {
        AgentConfig {
            command: "sh".to_string(),
            args: vec!["-lc".to_string(), "cat".to_string()],
            working_dir: working_dir.to_string_lossy().to_string(),
            env: HashMap::new(),
            inherit_env: Vec::new(),
        }
    }

    fn workspace_config(root: &Path, required: bool) -> WorkspaceConfig {
        WorkspaceConfig {
            root: Some(root.to_string_lossy().to_string()),
            required,
        }
    }

    fn with_home<T>(home_dir: &Path, f: impl FnOnce() -> T) -> T {
        let previous_home = std::env::var("HOME").ok();
        std::env::set_var("HOME", home_dir);
        let result = f();
        if let Some(home) = previous_home {
            std::env::set_var("HOME", home);
        } else {
            std::env::remove_var("HOME");
        }
        result
    }

    #[test]
    fn remove_if_same_handle_removes_matching_entry() {
        let expected = Arc::new(Mutex::new(1_u8));
        let mut map = HashMap::from([("thread".to_string(), Arc::clone(&expected))]);

        let removed = remove_if_same_handle(&mut map, "thread", &expected);

        assert!(removed.is_some());
        assert!(map.is_empty());
    }

    #[test]
    fn remove_if_same_handle_keeps_replaced_entry() {
        let stale = Arc::new(Mutex::new(1_u8));
        let fresh = Arc::new(Mutex::new(2_u8));
        let mut map = HashMap::from([("thread".to_string(), Arc::clone(&fresh))]);

        let removed = remove_if_same_handle(&mut map, "thread", &stale);

        assert!(removed.is_none());
        let current = map.get("thread").expect("entry should remain");
        assert!(Arc::ptr_eq(current, &fresh));
    }

    #[test]
    fn get_or_insert_gate_reuses_gate_for_same_thread() {
        let mut map = HashMap::new();

        let first = get_or_insert_gate(&mut map, "thread");
        let second = get_or_insert_gate(&mut map, "thread");

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn persisted_mapping_can_include_active_and_suspended_sessions() {
        let persisted = HashMap::from([
            ("active-thread".to_string(), "session-active".to_string()),
            (
                "suspended-thread".to_string(),
                "session-suspended".to_string(),
            ),
        ]);

        let serialized =
            serde_json::to_string_pretty(&persisted).expect("serialize persisted mapping");
        let roundtrip: HashMap<String, String> =
            serde_json::from_str(&serialized).expect("deserialize persisted mapping");

        assert_eq!(
            roundtrip.get("active-thread"),
            Some(&"session-active".to_string())
        );
        assert_eq!(
            roundtrip.get("suspended-thread"),
            Some(&"session-suspended".to_string())
        );
    }

    #[tokio::test]
    async fn resolve_working_dir_binds_required_workspace_create_without_rwlock_deadlock() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let project_dir = root.join("project");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(&root).expect("workspace root");
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, true),
            )
        });

        let result = tokio::time::timeout(
            Duration::from_millis(250),
            pool.resolve_working_dir(
                "discord:test-thread",
                Some(&WorkspaceRequest::Create("project".to_string())),
            ),
        )
        .await
        .expect("resolve_working_dir should not deadlock")
        .expect("workspace should resolve");

        let expected = project_dir.canonicalize().expect("canonical project dir");
        assert_eq!(result, expected.to_string_lossy());
    }

    #[tokio::test]
    async fn resolve_working_dir_rejects_missing_workspace_without_create() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let project_dir = root.join("missing-project");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(&root).expect("workspace root");
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, true),
            )
        });

        let err = pool
            .resolve_working_dir(
                "discord:test-thread",
                Some(&WorkspaceRequest::Existing("missing-project".to_string())),
            )
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("use [[ws:missing-project --create]]"));
        assert!(!project_dir.exists());
    }

    #[test]
    fn prepare_workspace_request_creates_then_normalizes_to_existing_relative_name() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let project_dir = root.join("project");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(&root).expect("workspace root");
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, true),
            )
        });

        let prepared = pool
            .prepare_workspace_request(Some(&WorkspaceRequest::Create("project".to_string())))
            .expect("workspace should create and normalize")
            .expect("request should stay present");

        assert!(project_dir.is_dir());
        assert_eq!(prepared, WorkspaceRequest::Existing("project".to_string()));
    }

    #[tokio::test]
    async fn preflighted_existing_workspace_resolves_to_canonical_cwd() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let project_dir = root.join("openab-lab");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(&project_dir).expect("project dir");
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, true),
            )
        });

        let prepared = pool
            .prepare_workspace_request(Some(&WorkspaceRequest::Existing("openab-lab".to_string())))
            .expect("workspace preflight should resolve")
            .expect("request should stay present");

        assert_eq!(
            prepared,
            WorkspaceRequest::Existing("openab-lab".to_string())
        );

        let cwd = pool
            .resolve_working_dir("discord:test-thread", Some(&prepared))
            .await
            .expect("preflighted workspace should create session cwd");

        let expected = project_dir.canonicalize().expect("canonical project dir");
        assert_eq!(cwd, expected.to_string_lossy());
    }

    #[test]
    fn prepare_workspace_request_rejects_existing_create() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let project_dir = root.join("project");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(&project_dir).expect("project dir");
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, true),
            )
        });

        let err = pool
            .prepare_workspace_request(Some(&WorkspaceRequest::Create("project".to_string())))
            .unwrap_err();

        assert!(err.to_string().contains("workspace already exists"));
        assert!(err.to_string().contains("use [[ws:project]]"));
    }

    #[test]
    fn prepare_workspace_request_ignores_ws_when_workspace_root_unset() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                WorkspaceConfig::default(),
            )
        });

        let prepared = pool
            .prepare_workspace_request(Some(&WorkspaceRequest::Existing("foo".to_string())))
            .expect("workspace feature should be disabled");
        assert_eq!(prepared, None);
    }

    #[test]
    fn prepare_workspace_request_requires_ws_when_required() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(&root).expect("workspace root");
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, true),
            )
        });

        let err = pool.prepare_workspace_request(None).unwrap_err();
        assert!(err.to_string().contains("missing workspace directive"));
    }

    #[test]
    fn prepare_workspace_request_allows_nested_workspace() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let project_dir = root.join("team").join("foo");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(&project_dir).expect("project dir");
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, false),
            )
        });

        let prepared = pool
            .prepare_workspace_request(Some(&WorkspaceRequest::Existing("team/foo".to_string())))
            .expect("nested workspace should resolve")
            .expect("request should stay present");

        assert_eq!(prepared, WorkspaceRequest::Existing("team/foo".to_string()));
    }

    #[test]
    fn prepare_workspace_request_rejects_absolute_and_parent_paths() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(&root).expect("workspace root");
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, false),
            )
        });

        let err = pool
            .prepare_workspace_request(Some(&WorkspaceRequest::Existing("/tmp/foo".to_string())))
            .unwrap_err();
        assert!(err.to_string().contains("relative path"));

        let err = pool
            .prepare_workspace_request(Some(&WorkspaceRequest::Existing("../foo".to_string())))
            .unwrap_err();
        assert!(err.to_string().contains("unsafe path components"));
    }

    #[cfg(unix)]
    #[test]
    fn prepare_workspace_request_rejects_symlink_parent_escape_before_create() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let outside = tmp.path().join("outside");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(&root).expect("workspace root");
        std::fs::create_dir_all(&outside).expect("outside dir");
        std::fs::create_dir_all(&home_dir).expect("home dir");
        symlink(&outside, root.join("link")).expect("symlink");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, false),
            )
        });

        let err = pool
            .prepare_workspace_request(Some(&WorkspaceRequest::Create("link/new".to_string())))
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("workspace parent is outside the workspace root"));
        assert!(!outside.join("new").exists());
    }
}
