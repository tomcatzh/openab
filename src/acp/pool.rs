use crate::acp::connection::AcpConnection;
use crate::acp::protocol::ConfigOption;
use crate::config::{AgentConfig, ThreadBindingConfig, WorkspaceConfig, WorkspaceRequest};
use crate::thread_binding::{
    PrimaryUpdateActor, ThreadBinding, ThreadBindingContext, ThreadBindingStore,
};
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
    thread_binding_store: Option<ThreadBindingStore>,
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
        thread_binding: ThreadBindingConfig,
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
        let thread_binding_store =
            ThreadBindingStore::from_config(thread_binding, workspace.root.as_deref());
        if let Some(store) = &thread_binding_store {
            if let Err(e) = store.initialize() {
                warn!(error = %e, "failed to initialize thread binding store");
            }
        }
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
            thread_binding_store,
        }
    }

    pub fn workspace_enabled(&self) -> bool {
        self.workspace.root.is_some()
    }

    pub fn current_agent_is_primary(
        &self,
        binding_context: &ThreadBindingContext,
        current_discord_user_id: &str,
    ) -> Result<bool> {
        let Some(store) = &self.thread_binding_store else {
            return Ok(false);
        };
        store.current_agent_is_primary(binding_context, current_discord_user_id)
    }

    pub fn message_mentions_profile_agent(
        &self,
        binding_context: &ThreadBindingContext,
        content: &str,
    ) -> Result<bool> {
        let Some(store) = &self.thread_binding_store else {
            return Ok(false);
        };
        store.message_mentions_profile_agent(binding_context, content)
    }

    pub fn handover_primary(
        &self,
        binding_context: &ThreadBindingContext,
        target_discord_user_id: &str,
        actor: PrimaryUpdateActor,
        message_id: &str,
    ) -> Result<ThreadBinding> {
        let Some(store) = &self.thread_binding_store else {
            return Err(anyhow!("thread binding store is not enabled"));
        };
        store.handover_primary(binding_context, target_discord_user_id, actor, message_id)
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

    fn validate_persisted_workdir(&self, existing: &str) -> Result<String> {
        let Some(root) = self.workspace_root() else {
            return Ok(existing.to_string());
        };
        let requested = PathBuf::from(existing);
        if !requested.is_dir() {
            return Err(anyhow!(
                "workspace no longer exists: {}; recreate it or start a new thread with [[ws:<name>]]",
                requested.display()
            ));
        }
        self.validate_existing_workspace_path(&requested, &root)
            .map_err(|e| anyhow!("persisted workspace is invalid: {existing}: {e}"))
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

    /// List immediate subdirectories of the workspace root for `/ws`
    /// slash-command autocomplete. Skips hidden entries, filters by
    /// case-insensitive substring, sorts (prefix matches first), and caps at
    /// `limit` (Discord allows at most 25 autocomplete choices).
    pub fn list_workspaces(&self, query: &str, limit: usize) -> Vec<String> {
        let Some(root) = self.workspace_root() else {
            return Vec::new();
        };
        let q = query.trim().to_lowercase();
        let mut names: Vec<String> = match std::fs::read_dir(&root) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| !n.starts_with('.'))
                .filter(|n| q.is_empty() || n.to_lowercase().contains(&q))
                .collect(),
            Err(_) => Vec::new(),
        };
        // Prefix matches first, then by name, so typing narrows intuitively.
        names.sort_by(|a, b| {
            let ap = a.to_lowercase().starts_with(&q);
            let bp = b.to_lowercase().starts_with(&q);
            bp.cmp(&ap).then_with(|| a.cmp(b))
        });
        names.truncate(limit);
        names
    }

    /// Validate a workspace request WITHOUT creating anything. Used by `/ws`
    /// before the Discord thread is created, so a bad name never leaves an
    /// orphan thread behind. Mirrors `validate_requested_workspace` minus the
    /// `create_dir_all` side effect.
    pub fn preflight_workspace_request(&self, request: &WorkspaceRequest) -> Result<()> {
        let Some(root) = self.workspace_root() else {
            return Ok(());
        };
        let rel = Self::validate_workspace_name(request.name())?;
        let requested = root.join(&rel);
        if !requested.starts_with(&root) {
            return Err(anyhow!("workspace is outside the workspace root"));
        }
        if requested.exists() {
            if request.creates_missing() {
                return Err(anyhow!(
                    "workspace already exists: {}; pick it from the list instead of passing create",
                    request.name()
                ));
            }
            self.validate_existing_workspace_path(&requested, &root)?;
            return Ok(());
        }
        if !request.creates_missing() {
            return Err(anyhow!(
                "workspace does not exist: {}; pass create: true to create it",
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
        Ok(())
    }

    /// Create/validate the requested workspace and bind it to a freshly-created
    /// thread up front. `/ws` creates the Discord thread before any agent turn
    /// runs, so the binding cannot wait for `resolve_working_dir`. Persists the
    /// per-bot `thread_id -> cwd` map (the source of truth for single-bot
    /// follow-ups) and best-effort records the shared ChannelProfile binding so
    /// other bots can inherit the cwd. Returns the resolved absolute cwd.
    pub async fn bind_thread_workspace(
        &self,
        thread_id: &str,
        request: &WorkspaceRequest,
        binding_context: Option<&ThreadBindingContext>,
    ) -> Result<String> {
        let cwd = self.validate_requested_workspace(request)?;
        {
            let mut state = self.state.write().await;
            if let Some(existing) = state.workdirs.get(thread_id) {
                let existing = existing.clone();
                if existing == cwd {
                    return Ok(cwd);
                }
                return Err(anyhow!(
                    "thread already has workspace {existing}; start a new thread to change workspace"
                ));
            }
            state.workdirs.insert(thread_id.to_string(), cwd.clone());
            self.save_workdir_mapping(&state.workdirs);
        }
        if let (Some(store), Some(ctx)) = (&self.thread_binding_store, binding_context) {
            if let Err(e) = store.record_binding(ctx, request.name(), &cwd) {
                warn!(
                    error = %e,
                    "failed to record shared thread binding for /ws; \
                     continuing with per-bot binding"
                );
            }
        }
        info!(thread_id, working_dir = %cwd, "bound thread workspace via /ws");
        Ok(cwd)
    }

    async fn resolve_working_dir(
        &self,
        thread_id: &str,
        workspace_request: Option<&WorkspaceRequest>,
        binding_context: Option<&ThreadBindingContext>,
    ) -> Result<String> {
        if self.workspace_enabled() {
            let existing_workdir = {
                let state = self.state.read().await;
                state.workdirs.get(thread_id).cloned()
            };

            if let Some(existing) = existing_workdir {
                let existing = self.validate_persisted_workdir(&existing)?;
                if workspace_request.is_some() {
                    let requested = self.validate_requested_workspace(
                        workspace_request.expect("checked workspace_request is some"),
                    )?;
                    if requested == existing {
                        return Ok(existing);
                    }
                    return Err(anyhow!("thread already has workspace {existing}; start a new thread to change workspace"));
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
                if let (Some(store), Some(ctx)) = (&self.thread_binding_store, binding_context) {
                    store.record_binding(ctx, requested.name(), &cwd)?;
                }
                info!(thread_id, working_dir = %cwd, "bound thread workspace");
                return Ok(cwd);
            }

            if let (Some(store), Some(ctx)) = (&self.thread_binding_store, binding_context) {
                if let Some(binding) = store.lookup_binding(ctx)? {
                    let cwd = self.validate_persisted_workdir(&binding.resolved_cwd)?;
                    let mut state = self.state.write().await;
                    state.workdirs.insert(thread_id.to_string(), cwd.clone());
                    self.save_workdir_mapping(&state.workdirs);
                    info!(thread_id, working_dir = %cwd, "inherited shared thread workspace");
                    return Ok(cwd);
                }
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

    pub async fn working_dir_for_thread(&self, thread_id: &str) -> Result<String> {
        if let Some(existing) = {
            let state = self.state.read().await;
            state.workdirs.get(thread_id).cloned()
        } {
            return self.validate_persisted_workdir(&existing);
        }

        if self.per_thread_workdir {
            let safe = Self::safe_thread_component(thread_id)?;
            let dir = PathBuf::from(&self.config.working_dir)
                .join("sessions")
                .join(safe);
            return Ok(dir.to_string_lossy().to_string());
        }

        Ok(self.config.working_dir.clone())
    }

    pub async fn get_or_create(
        &self,
        thread_id: &str,
        workspace_request: Option<&WorkspaceRequest>,
        binding_context: Option<&ThreadBindingContext>,
    ) -> Result<()> {
        let create_gate = {
            let mut state = self.state.write().await;
            get_or_insert_gate(&mut state.creating, thread_id)
        };
        let _create_guard = create_gate.lock().await;
        let working_dir = self
            .resolve_working_dir(thread_id, workspace_request, binding_context)
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
    use crate::config::{
        AgentConfig, ChannelProfileConfig, ThreadBindingConfig, WorkspaceConfig, WorkspaceRequest,
    };
    use crate::thread_binding::ThreadBindingContext;
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

    fn thread_binding_config() -> ThreadBindingConfig {
        ThreadBindingConfig::default()
    }

    fn thread_binding_config_with_profile(
        root: &Path,
        store_dir: &Path,
        agent_name: &str,
    ) -> ThreadBindingConfig {
        ThreadBindingConfig {
            enabled: true,
            agent_name: Some(agent_name.to_string()),
            store_dir: Some(store_dir.to_string_lossy().to_string()),
            channel_profiles: vec![ChannelProfileConfig {
                profile_id: "software-dev".to_string(),
                guild_id: "111111111111111111".to_string(),
                parent_channel_id: "222222222222222222".to_string(),
                work_domain: "software-development".to_string(),
                workspace_root: root.to_string_lossy().to_string(),
                allowed_agents: vec!["codex".to_string(), "codex-reviewer".to_string()],
                agents: Vec::new(),
            }],
        }
    }

    fn binding_context(thread_id: &str) -> ThreadBindingContext {
        ThreadBindingContext {
            platform: "discord".to_string(),
            guild_id: Some("111111111111111111".to_string()),
            parent_channel_id: Some("222222222222222222".to_string()),
            thread_id: thread_id.to_string(),
            created_by_user_id: Some("333333333333333333".to_string()),
            trigger_message_id: Some("444444444444444444".to_string()),
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
                thread_binding_config(),
            )
        });

        let result = tokio::time::timeout(
            Duration::from_millis(250),
            pool.resolve_working_dir(
                "discord:test-thread",
                Some(&WorkspaceRequest::Create("project".to_string())),
                None,
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
                thread_binding_config(),
            )
        });

        let err = pool
            .resolve_working_dir(
                "discord:test-thread",
                Some(&WorkspaceRequest::Existing("missing-project".to_string())),
                None,
            )
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("use [[ws:missing-project --create]]"));
        assert!(!project_dir.exists());
    }

    #[tokio::test]
    async fn resolve_working_dir_rejects_deleted_persisted_workspace() {
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
                thread_binding_config(),
            )
        });

        let thread_id = "discord:test-thread";
        pool.resolve_working_dir(
            thread_id,
            Some(&WorkspaceRequest::Existing("project".to_string())),
            None,
        )
        .await
        .expect("workspace should bind");

        std::fs::remove_dir(&project_dir).expect("delete project workspace");

        let err = pool
            .resolve_working_dir(thread_id, None, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("workspace no longer exists"));
        assert!(err.to_string().contains("project"));
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
                thread_binding_config(),
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
                thread_binding_config(),
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
            .resolve_working_dir("discord:test-thread", Some(&prepared), None)
            .await
            .expect("preflighted workspace should create session cwd");

        let expected = project_dir.canonicalize().expect("canonical project dir");
        assert_eq!(cwd, expected.to_string_lossy());
    }

    #[test]
    fn list_workspaces_filters_hidden_and_orders_prefix_first() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let home_dir = tmp.path().join("home");
        for d in ["openab-deploy", "openab-lab", "httpping", ".git", ".openab"] {
            std::fs::create_dir_all(root.join(d)).expect("mkdir");
        }
        std::fs::write(root.join("a-file"), "x").expect("file"); // non-dir ignored
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, true),
                thread_binding_config(),
            )
        });

        // No query → all visible dirs, sorted, hidden + files excluded.
        let all = pool.list_workspaces("", 25);
        assert_eq!(all, vec!["httpping", "openab-deploy", "openab-lab"]);

        // Query narrows; substring "lab" matches openab-lab.
        assert_eq!(pool.list_workspaces("lab", 25), vec!["openab-lab"]);

        // Prefix matches sort ahead of mid-string matches.
        std::fs::create_dir_all(root.join("zz-openab")).expect("mkdir");
        let q = pool.list_workspaces("openab", 25);
        assert_eq!(q[0], "openab-deploy");
        assert_eq!(q[1], "openab-lab");
        assert!(q.contains(&"zz-openab".to_string()));
        assert!(q.iter().position(|n| n == "zz-openab").unwrap() > 1);

        // Limit is honored.
        assert_eq!(pool.list_workspaces("", 1).len(), 1);
    }

    #[test]
    fn preflight_workspace_request_validates_without_side_effects() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(root.join("project")).expect("project dir");
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, true),
                thread_binding_config(),
            )
        });

        // Existing without create → ok.
        pool.preflight_workspace_request(&WorkspaceRequest::Existing("project".to_string()))
            .expect("existing should pass");

        // Missing without create → error, no directory created.
        let err = pool
            .preflight_workspace_request(&WorkspaceRequest::Existing("missing".to_string()))
            .unwrap_err();
        assert!(err.to_string().contains("does not exist"));
        assert!(!root.join("missing").exists());

        // Missing with create → ok, still no side effect (preflight does not mkdir).
        pool.preflight_workspace_request(&WorkspaceRequest::Create("fresh".to_string()))
            .expect("create preflight should pass");
        assert!(!root.join("fresh").exists());

        // Existing with create → error (would clobber intent).
        let err = pool
            .preflight_workspace_request(&WorkspaceRequest::Create("project".to_string()))
            .unwrap_err();
        assert!(err.to_string().contains("already exists"));

        // Traversal rejected.
        assert!(pool
            .preflight_workspace_request(&WorkspaceRequest::Existing("../escape".to_string()))
            .is_err());
    }

    #[tokio::test]
    async fn bind_thread_workspace_creates_dir_and_is_idempotent() {
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
                thread_binding_config(),
            )
        });

        let thread_id = "discord:ws-thread";
        let cwd = pool
            .bind_thread_workspace(
                thread_id,
                &WorkspaceRequest::Create("fresh".to_string()),
                None,
            )
            .await
            .expect("bind should create and resolve");
        let expected = root.join("fresh").canonicalize().expect("canonical");
        assert_eq!(cwd, expected.to_string_lossy());
        assert!(root.join("fresh").is_dir());

        // A later turn with no directive resolves the same bound cwd.
        let resolved = pool
            .resolve_working_dir(thread_id, None, None)
            .await
            .expect("follow-up resolves persisted cwd");
        assert_eq!(resolved, cwd);

        // Re-binding the same workspace is idempotent.
        let again = pool
            .bind_thread_workspace(
                thread_id,
                &WorkspaceRequest::Existing("fresh".to_string()),
                None,
            )
            .await
            .expect("idempotent rebind");
        assert_eq!(again, cwd);

        // Binding a different workspace to the same thread is rejected.
        std::fs::create_dir_all(root.join("other")).expect("other dir");
        let err = pool
            .bind_thread_workspace(
                thread_id,
                &WorkspaceRequest::Existing("other".to_string()),
                None,
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("already has workspace"));
    }

    #[tokio::test]
    async fn thread_binding_inherits_cwd_across_independent_homes() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let project_dir = root.join("openab-lab");
        let store_dir = root.join(".openab");
        let home_a = tmp.path().join("home-a");
        let home_b = tmp.path().join("home-b");
        std::fs::create_dir_all(&project_dir).expect("project dir");
        std::fs::create_dir_all(&home_a).expect("home a");
        std::fs::create_dir_all(&home_b).expect("home b");

        let pool_a = with_home(&home_a, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, true),
                thread_binding_config_with_profile(&root, &store_dir, "codex"),
            )
        });
        let pool_b = with_home(&home_b, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, true),
                thread_binding_config_with_profile(&root, &store_dir, "codex-reviewer"),
            )
        });
        let ctx = binding_context("555555555555555555");
        let thread_key = format!("discord:{}", ctx.thread_id);

        let bound = pool_a
            .resolve_working_dir(
                &thread_key,
                Some(&WorkspaceRequest::Existing("openab-lab".to_string())),
                Some(&ctx),
            )
            .await
            .expect("bot a should bind workspace");
        let inherited = pool_b
            .resolve_working_dir(&thread_key, None, Some(&ctx))
            .await
            .expect("bot b should inherit shared binding");

        let expected = project_dir.canonicalize().expect("canonical project dir");
        assert_eq!(bound, expected.to_string_lossy());
        assert_eq!(inherited, expected.to_string_lossy());

        let private_map =
            std::fs::read_to_string(home_b.join(".openab").join("thread_workdir_map.json"))
                .expect("bot b private workdir map");
        let workdirs: HashMap<String, String> =
            serde_json::from_str(&private_map).expect("parse workdir map");
        assert_eq!(workdirs.get(&thread_key), Some(&inherited));
    }

    #[tokio::test]
    async fn repeated_same_workspace_is_allowed_but_workspace_change_is_rejected() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("workspaces");
        let project_dir = root.join("openab-lab");
        let other_dir = root.join("other");
        let home_dir = tmp.path().join("home");
        std::fs::create_dir_all(&project_dir).expect("project dir");
        std::fs::create_dir_all(&other_dir).expect("other dir");
        std::fs::create_dir_all(&home_dir).expect("home dir");

        let pool = with_home(&home_dir, || {
            SessionPool::new(
                agent_config(tmp.path()),
                1,
                false,
                workspace_config(&root, true),
                thread_binding_config(),
            )
        });
        let thread_key = "discord:test-thread";
        let first = pool
            .resolve_working_dir(
                thread_key,
                Some(&WorkspaceRequest::Existing("openab-lab".to_string())),
                None,
            )
            .await
            .expect("initial bind");
        let repeated = pool
            .resolve_working_dir(
                thread_key,
                Some(&WorkspaceRequest::Existing("openab-lab".to_string())),
                None,
            )
            .await
            .expect("same workspace redeclaration should pass");

        assert_eq!(repeated, first);

        let err = pool
            .resolve_working_dir(
                thread_key,
                Some(&WorkspaceRequest::Existing("other".to_string())),
                None,
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("thread already has workspace"));
        assert!(err.to_string().contains("openab-lab"));
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
                thread_binding_config(),
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
                thread_binding_config(),
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
                thread_binding_config(),
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
                thread_binding_config(),
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
                thread_binding_config(),
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
                thread_binding_config(),
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
