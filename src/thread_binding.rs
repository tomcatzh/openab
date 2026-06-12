use crate::config::{ChannelProfileAgentConfig, ChannelProfileConfig, ThreadBindingConfig};
use anyhow::{anyhow, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

const CHANNEL_PROFILES_FILE: &str = "channel-profiles.json";
const THREAD_BINDINGS_FILE: &str = "thread-bindings.json";
const THREAD_BINDINGS_LOCK: &str = "thread-bindings.lock";
// A healthy holder keeps the lock only for one read-modify-write of a small
// JSON file (well under a second). A lockfile older than this is presumed to
// have been orphaned by a writer that died mid-update (e.g. an OOM kill on the
// shared workspace PVC) and is reclaimed so writes do not wedge permanently.
const LOCK_STALE_SECS: u64 = 30;
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChannelProfile {
    pub profile_id: String,
    pub guild_id: String,
    pub parent_channel_id: String,
    pub work_domain: String,
    pub workspace_root: String,
    pub allowed_agents: Vec<String>,
    pub agents: Vec<ChannelProfileAgent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChannelProfileAgent {
    pub name: String,
    pub aliases: Vec<String>,
    pub discord_user_id: Option<String>,
    pub discord_role_id: Option<String>,
    pub handoff_hint: Option<String>,
}

impl From<ChannelProfileAgentConfig> for ChannelProfileAgent {
    fn from(value: ChannelProfileAgentConfig) -> Self {
        Self {
            name: value.name,
            aliases: value.aliases,
            discord_user_id: value.discord_user_id,
            discord_role_id: value.discord_role_id,
            handoff_hint: value.handoff_hint,
        }
    }
}

impl ChannelProfileAgent {
    fn name_only(name: &str) -> Self {
        Self {
            name: name.to_string(),
            aliases: Vec::new(),
            discord_user_id: None,
            discord_role_id: None,
            handoff_hint: None,
        }
    }
}

impl From<ChannelProfileConfig> for ChannelProfile {
    fn from(value: ChannelProfileConfig) -> Self {
        let mut agents: Vec<ChannelProfileAgent> = value
            .agents
            .into_iter()
            .map(ChannelProfileAgent::from)
            .collect();
        let mut allowed_agents = value.allowed_agents;
        if allowed_agents.is_empty() {
            allowed_agents = agents.iter().map(|agent| agent.name.clone()).collect();
        }
        if agents.is_empty() {
            agents = allowed_agents
                .iter()
                .map(|agent| ChannelProfileAgent::name_only(agent))
                .collect();
        }
        Self {
            profile_id: value.profile_id,
            guild_id: value.guild_id,
            parent_channel_id: value.parent_channel_id,
            work_domain: value.work_domain,
            workspace_root: value.workspace_root,
            allowed_agents,
            agents,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThreadBinding {
    pub platform: String,
    pub guild_id: String,
    pub parent_channel_id: String,
    pub thread_id: String,
    pub workspace_name: String,
    pub resolved_cwd: String,
    pub created_by_agent: String,
    pub created_by_user_id: String,
    pub trigger_message_id: String,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_bot: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_bot_discord_user_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_updated_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_updated_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_update_message_id: Option<String>,
}

impl ThreadBinding {
    pub fn effective_primary_bot(&self) -> &str {
        self.primary_bot
            .as_deref()
            .unwrap_or(&self.created_by_agent)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimaryUpdateActor {
    Human { user_id: String },
    Bot { user_id: String },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct ChannelProfilesFile {
    schema: String,
    profiles: Vec<ChannelProfile>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct ThreadBindingsFile {
    schema: String,
    bindings: BTreeMap<String, ThreadBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadBindingContext {
    pub platform: String,
    pub guild_id: Option<String>,
    pub parent_channel_id: Option<String>,
    pub thread_id: String,
    pub created_by_user_id: Option<String>,
    pub trigger_message_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ThreadBindingStore {
    store_dir: PathBuf,
    agent_name: String,
    profiles: Vec<ChannelProfile>,
}

#[derive(Debug)]
struct LockGuard {
    path: PathBuf,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if let Err(e) = fs::remove_file(&self.path) {
            if e.kind() != ErrorKind::NotFound {
                warn!(path = %self.path.display(), error = %e, "failed to remove thread binding lock");
            }
        }
    }
}

impl ThreadBindingStore {
    pub fn from_config(config: ThreadBindingConfig, workspace_root: Option<&str>) -> Option<Self> {
        if !config.enabled {
            return None;
        }
        let store_dir = config
            .store_dir
            .map(PathBuf::from)
            .or_else(|| workspace_root.map(|root| PathBuf::from(root).join(".openab")))?;
        let profiles = config
            .channel_profiles
            .into_iter()
            .map(ChannelProfile::from)
            .collect();
        Some(Self {
            store_dir,
            agent_name: config.agent_name.unwrap_or_else(|| "default".to_string()),
            profiles,
        })
    }

    pub fn initialize(&self) -> Result<()> {
        fs::create_dir_all(&self.store_dir)?;
        if !self.profiles.is_empty() {
            let file = ChannelProfilesFile {
                schema: "openab.channel_profiles.v1".to_string(),
                profiles: self.profiles.clone(),
            };
            self.write_json(&self.channel_profiles_path(), &file)?;
        }
        if !self.thread_bindings_path().exists() {
            let _guard = self.acquire_lock()?;
            if !self.thread_bindings_path().exists() {
                self.write_bindings_file(&ThreadBindingsFile {
                    schema: "openab.thread_bindings.v1".to_string(),
                    bindings: BTreeMap::new(),
                })?;
            }
        }
        Ok(())
    }

    pub fn record_binding(
        &self,
        ctx: &ThreadBindingContext,
        workspace_name: &str,
        resolved_cwd: &str,
    ) -> Result<()> {
        self.initialize()?;
        let profile = self.matching_profile(ctx)?;
        self.validate_agent_allowed(&profile)?;
        self.validate_cwd(&profile, resolved_cwd)?;
        let created_at = Utc::now().to_rfc3339();
        let primary_agent = profile
            .agents
            .iter()
            .find(|agent| agent.name == self.agent_name);

        let binding = ThreadBinding {
            platform: ctx.platform.clone(),
            guild_id: ctx
                .guild_id
                .clone()
                .ok_or_else(|| anyhow!("missing guild id for thread binding"))?,
            parent_channel_id: ctx
                .parent_channel_id
                .clone()
                .ok_or_else(|| anyhow!("missing parent channel id for thread binding"))?,
            thread_id: ctx.thread_id.clone(),
            workspace_name: workspace_name.to_string(),
            resolved_cwd: resolved_cwd.to_string(),
            created_by_agent: self.agent_name.clone(),
            created_by_user_id: ctx.created_by_user_id.clone().unwrap_or_default(),
            trigger_message_id: ctx.trigger_message_id.clone().unwrap_or_default(),
            created_at: created_at.clone(),
            primary_bot: Some(self.agent_name.clone()),
            primary_bot_discord_user_id: primary_agent
                .and_then(|agent| agent.discord_user_id.clone()),
            primary_updated_by: Some(format!("agent:{}", self.agent_name)),
            primary_updated_at: Some(created_at),
            primary_update_message_id: ctx.trigger_message_id.clone(),
        };

        let _guard = self.acquire_lock()?;
        let mut file = self.load_bindings_file()?;
        let key = Self::binding_key(&binding.platform, &binding.thread_id);
        if let Some(existing) = file.bindings.get(&key) {
            if existing.resolved_cwd != binding.resolved_cwd {
                return Err(anyhow!(
                    "thread already has workspace {}; start a new thread to change workspace",
                    existing.resolved_cwd
                ));
            }
            return Ok(());
        }
        file.bindings.insert(key, binding);
        self.write_bindings_file(&file)?;
        info!(thread_id = %ctx.thread_id, resolved_cwd, "recorded shared thread binding");
        Ok(())
    }

    pub fn lookup_binding(&self, ctx: &ThreadBindingContext) -> Result<Option<ThreadBinding>> {
        self.initialize()?;
        let profile = self.matching_profile(ctx)?;
        self.validate_agent_allowed(&profile)?;
        let file = self.load_bindings_file()?;
        let key = Self::binding_key(&ctx.platform, &ctx.thread_id);
        let Some(binding) = file.bindings.get(&key).cloned() else {
            return Ok(None);
        };
        if binding.parent_channel_id != profile.parent_channel_id {
            return Err(anyhow!("thread binding parent channel mismatch"));
        }
        self.validate_cwd(&profile, &binding.resolved_cwd)?;
        Ok(Some(self.with_primary_defaults(binding, &profile)))
    }

    pub fn current_agent_is_primary(
        &self,
        ctx: &ThreadBindingContext,
        current_discord_user_id: &str,
    ) -> Result<bool> {
        self.initialize()?;
        let profile = self.matching_profile(ctx)?;
        self.validate_agent_allowed(&profile)?;
        if let Some(agent) = profile
            .agents
            .iter()
            .find(|agent| agent.name == self.agent_name)
        {
            if agent
                .discord_user_id
                .as_deref()
                .is_some_and(|id| id != current_discord_user_id)
            {
                return Ok(false);
            }
        }
        let file = self.load_bindings_file()?;
        let key = Self::binding_key(&ctx.platform, &ctx.thread_id);
        let Some(binding) = file.bindings.get(&key).cloned() else {
            return Ok(false);
        };
        if binding.parent_channel_id != profile.parent_channel_id {
            return Err(anyhow!("thread binding parent channel mismatch"));
        }
        self.validate_cwd(&profile, &binding.resolved_cwd)?;
        let binding = self.with_primary_defaults(binding, &profile);
        Ok(binding.effective_primary_bot() == self.agent_name)
    }

    pub fn message_mentions_profile_agent(
        &self,
        ctx: &ThreadBindingContext,
        content: &str,
    ) -> Result<bool> {
        self.initialize()?;
        let profile = self.matching_profile(ctx)?;
        for agent in &profile.agents {
            if let Some(user_id) = &agent.discord_user_id {
                if content.contains(&format!("<@{user_id}>"))
                    || content.contains(&format!("<@!{user_id}>"))
                {
                    return Ok(true);
                }
            }
            if let Some(role_id) = &agent.discord_role_id {
                if content.contains(&format!("<@&{role_id}>")) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    pub fn handover_primary(
        &self,
        ctx: &ThreadBindingContext,
        target_discord_user_id: &str,
        actor: PrimaryUpdateActor,
        message_id: &str,
    ) -> Result<ThreadBinding> {
        self.initialize()?;
        let profile = self.matching_profile(ctx)?;
        self.validate_agent_allowed(&profile)?;
        let target_agent = profile
            .agents
            .iter()
            .find(|agent| agent.discord_user_id.as_deref() == Some(target_discord_user_id))
            .ok_or_else(|| anyhow!("handover target is not in ChannelProfile"))?;
        if target_agent.name != self.agent_name {
            return Err(anyhow!("handover target does not match this agent"));
        }
        if !profile
            .allowed_agents
            .iter()
            .any(|agent| agent == &target_agent.name)
        {
            return Err(anyhow!("handover target is not allowed in ChannelProfile"));
        }

        let _guard = self.acquire_lock()?;
        let mut file = self.load_bindings_file()?;
        let key = Self::binding_key(&ctx.platform, &ctx.thread_id);
        let Some(existing) = file.bindings.get(&key).cloned() else {
            return Err(anyhow!("thread is not bound to a workspace"));
        };
        if existing.parent_channel_id != profile.parent_channel_id {
            return Err(anyhow!("thread binding parent channel mismatch"));
        }
        self.validate_cwd(&profile, &existing.resolved_cwd)?;
        let mut binding = self.with_primary_defaults(existing, &profile);

        let updated_by = match actor {
            PrimaryUpdateActor::Human { user_id } => format!("user:{user_id}"),
            PrimaryUpdateActor::Bot { user_id } => {
                let source_agent = profile
                    .agents
                    .iter()
                    .find(|agent| agent.discord_user_id.as_deref() == Some(user_id.as_str()))
                    .ok_or_else(|| anyhow!("handover author bot is not in ChannelProfile"))?;
                if source_agent.name != binding.effective_primary_bot() {
                    return Err(anyhow!("only the current primary bot can hand over"));
                }
                format!("agent:{}", source_agent.name)
            }
        };

        let updated_at = Utc::now().to_rfc3339();
        binding.primary_bot = Some(target_agent.name.clone());
        binding.primary_bot_discord_user_id = target_agent.discord_user_id.clone();
        binding.primary_updated_by = Some(updated_by);
        binding.primary_updated_at = Some(updated_at);
        binding.primary_update_message_id = Some(message_id.to_string());
        file.bindings.insert(key, binding.clone());
        self.write_bindings_file(&file)?;
        info!(
            thread_id = %ctx.thread_id,
            primary_bot = %target_agent.name,
            "updated thread primary bot"
        );
        Ok(binding)
    }

    fn matching_profile(&self, ctx: &ThreadBindingContext) -> Result<ChannelProfile> {
        let profiles = self.load_profiles()?;
        let guild_id = ctx
            .guild_id
            .as_deref()
            .ok_or_else(|| anyhow!("missing guild id for channel profile lookup"))?;
        let parent_channel_id = ctx
            .parent_channel_id
            .as_deref()
            .ok_or_else(|| anyhow!("missing parent channel id for channel profile lookup"))?;
        profiles
            .into_iter()
            .find(|profile| {
                profile.guild_id == guild_id && profile.parent_channel_id == parent_channel_id
            })
            .ok_or_else(|| anyhow!("no ChannelProfile matches this Discord channel"))
    }

    fn validate_agent_allowed(&self, profile: &ChannelProfile) -> Result<()> {
        if profile
            .allowed_agents
            .iter()
            .any(|agent| agent == &self.agent_name)
        {
            Ok(())
        } else {
            Err(anyhow!(
                "agent {} is not allowed in ChannelProfile {}",
                self.agent_name,
                profile.profile_id
            ))
        }
    }

    fn validate_cwd(&self, profile: &ChannelProfile, cwd: &str) -> Result<()> {
        let root = fs::canonicalize(&profile.workspace_root).map_err(|e| {
            anyhow!(
                "failed to resolve ChannelProfile workspace root {}: {e}",
                profile.workspace_root
            )
        })?;
        let cwd = fs::canonicalize(cwd)
            .map_err(|e| anyhow!("failed to resolve thread binding cwd {cwd}: {e}"))?;
        if !cwd.is_dir() {
            return Err(anyhow!("thread binding cwd is not a directory"));
        }
        if cwd.starts_with(&root) {
            Ok(())
        } else {
            Err(anyhow!(
                "thread binding cwd is outside the ChannelProfile workspace root"
            ))
        }
    }

    fn with_primary_defaults(
        &self,
        mut binding: ThreadBinding,
        profile: &ChannelProfile,
    ) -> ThreadBinding {
        if binding.primary_bot.is_none() {
            binding.primary_bot = Some(binding.created_by_agent.clone());
        }
        if binding.primary_bot_discord_user_id.is_none() {
            let primary = binding.effective_primary_bot();
            binding.primary_bot_discord_user_id = profile
                .agents
                .iter()
                .find(|agent| agent.name == primary)
                .and_then(|agent| agent.discord_user_id.clone());
        }
        binding
    }

    fn load_profiles(&self) -> Result<Vec<ChannelProfile>> {
        if self.channel_profiles_path().exists() {
            let data = fs::read_to_string(self.channel_profiles_path())?;
            let file: ChannelProfilesFile = serde_json::from_str(&data)?;
            return Ok(file.profiles);
        }
        Ok(self.profiles.clone())
    }

    fn load_bindings_file(&self) -> Result<ThreadBindingsFile> {
        match fs::read_to_string(self.thread_bindings_path()) {
            Ok(data) => Ok(serde_json::from_str(&data)?),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(ThreadBindingsFile {
                schema: "openab.thread_bindings.v1".to_string(),
                bindings: BTreeMap::new(),
            }),
            Err(e) => Err(e.into()),
        }
    }

    fn write_bindings_file(&self, file: &ThreadBindingsFile) -> Result<()> {
        self.write_json(&self.thread_bindings_path(), file)
    }

    fn write_json<T: Serialize>(&self, path: &Path, value: &T) -> Result<()> {
        fs::create_dir_all(&self.store_dir)?;
        let data = serde_json::to_string_pretty(value)?;
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let counter = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp = path.with_extension(format!("{}.{}.{}.tmp", std::process::id(), counter, unique));
        fs::write(&tmp, data)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    fn acquire_lock(&self) -> Result<LockGuard> {
        self.acquire_lock_with(Duration::from_secs(LOCK_STALE_SECS))
    }

    fn acquire_lock_with(&self, stale_after: Duration) -> Result<LockGuard> {
        fs::create_dir_all(&self.store_dir)?;
        let path = self.lock_path();
        for _ in 0..250 {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(_) => return Ok(LockGuard { path }),
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    Self::try_reclaim_stale_lock(&path, stale_after);
                    thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(e.into()),
            }
        }
        Err(anyhow!("timed out acquiring thread binding lock"))
    }

    /// Reclaim a lockfile left behind by a writer that died mid-update. The
    /// steal is done via an atomic rename: the rename source exists exactly
    /// once, so two racing reclaimers can never both delete a lock — the loser
    /// gets `NotFound` and simply retries `create_new`. A freshly created lock
    /// can never be stolen because a live holder's lockfile is younger than
    /// `stale_after`, and `create_new` keeps the source present until the
    /// winner renames it away.
    fn try_reclaim_stale_lock(path: &Path, stale_after: Duration) {
        let Ok(modified) = fs::metadata(path).and_then(|m| m.modified()) else {
            return;
        };
        let is_stale = SystemTime::now()
            .duration_since(modified)
            .map(|age| age >= stale_after)
            .unwrap_or(false);
        if !is_stale {
            return;
        }
        let counter = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let steal_path = path.with_extension(format!("steal.{}.{}", std::process::id(), counter));
        if fs::rename(path, &steal_path).is_ok() {
            let _ = fs::remove_file(&steal_path);
            warn!(path = %path.display(), "reclaimed stale thread binding lock");
        }
    }

    fn binding_key(platform: &str, thread_id: &str) -> String {
        format!("{platform}:{thread_id}")
    }

    fn channel_profiles_path(&self) -> PathBuf {
        self.store_dir.join(CHANNEL_PROFILES_FILE)
    }

    fn thread_bindings_path(&self) -> PathBuf {
        self.store_dir.join(THREAD_BINDINGS_FILE)
    }

    fn lock_path(&self) -> PathBuf {
        self.store_dir.join(THREAD_BINDINGS_LOCK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn profile(root: &Path) -> ChannelProfileConfig {
        ChannelProfileConfig {
            profile_id: "dev".to_string(),
            guild_id: "111111111111111111".to_string(),
            parent_channel_id: "222222222222222222".to_string(),
            work_domain: "development".to_string(),
            workspace_root: root.to_string_lossy().to_string(),
            allowed_agents: vec!["codex".to_string(), "reviewer".to_string()],
            agents: vec![
                ChannelProfileAgentConfig {
                    name: "codex".to_string(),
                    aliases: vec!["A".to_string()],
                    discord_user_id: Some("9001".to_string()),
                    discord_role_id: None,
                    handoff_hint: None,
                },
                ChannelProfileAgentConfig {
                    name: "reviewer".to_string(),
                    aliases: vec!["B".to_string()],
                    discord_user_id: Some("9002".to_string()),
                    discord_role_id: Some("9902".to_string()),
                    handoff_hint: None,
                },
            ],
        }
    }

    fn config(root: &Path, store_dir: &Path, agent_name: &str) -> ThreadBindingConfig {
        ThreadBindingConfig {
            enabled: true,
            agent_name: Some(agent_name.to_string()),
            store_dir: Some(store_dir.to_string_lossy().to_string()),
            channel_profiles: vec![profile(root)],
        }
    }

    fn ctx(thread_id: &str) -> ThreadBindingContext {
        ThreadBindingContext {
            platform: "discord".to_string(),
            guild_id: Some("111111111111111111".to_string()),
            parent_channel_id: Some("222222222222222222".to_string()),
            thread_id: thread_id.to_string(),
            created_by_user_id: Some("333333333333333333".to_string()),
            trigger_message_id: Some("444444444444444444".to_string()),
        }
    }

    #[test]
    fn serde_round_trips_channel_profile_and_thread_binding() {
        let tmp = TempDir::new().expect("tmp");
        let profile = ChannelProfile::from(profile(tmp.path()));
        assert_eq!(
            profile
                .agents
                .iter()
                .map(|agent| agent.name.as_str())
                .collect::<Vec<_>>(),
            vec!["codex", "reviewer"]
        );

        let binding = ThreadBinding {
            platform: "discord".to_string(),
            guild_id: "111111111111111111".to_string(),
            parent_channel_id: "222222222222222222".to_string(),
            thread_id: "555555555555555555".to_string(),
            workspace_name: "openab-lab".to_string(),
            resolved_cwd: "/workspace/openab-lab".to_string(),
            created_by_agent: "codex".to_string(),
            created_by_user_id: "333333333333333333".to_string(),
            trigger_message_id: "444444444444444444".to_string(),
            created_at: "2026-06-04T00:00:00Z".to_string(),
            primary_bot: Some("codex".to_string()),
            primary_bot_discord_user_id: Some("9001".to_string()),
            primary_updated_by: Some("agent:codex".to_string()),
            primary_updated_at: Some("2026-06-04T00:00:00Z".to_string()),
            primary_update_message_id: Some("444444444444444444".to_string()),
        };
        let data = serde_json::to_string(&binding).expect("serialize binding");
        let restored: ThreadBinding = serde_json::from_str(&data).expect("deserialize binding");
        assert_eq!(restored, binding);
    }

    #[test]
    fn serde_reads_legacy_binding_without_primary_fields() {
        let data = r#"{
          "platform": "discord",
          "guild_id": "111111111111111111",
          "parent_channel_id": "222222222222222222",
          "thread_id": "555555555555555555",
          "workspace_name": "openab-lab",
          "resolved_cwd": "/workspace/openab-lab",
          "created_by_agent": "codex",
          "created_by_user_id": "333333333333333333",
          "trigger_message_id": "444444444444444444",
          "created_at": "2026-06-04T00:00:00Z"
        }"#;
        let restored: ThreadBinding = serde_json::from_str(data).expect("deserialize legacy");
        assert_eq!(restored.effective_primary_bot(), "codex");
        assert_eq!(restored.primary_bot, None);
    }

    #[test]
    fn records_and_reads_binding_from_workspace_store() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let project = root.join("openab-lab");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&project).expect("project");
        let store = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store");

        store
            .record_binding(
                &ctx("555555555555555555"),
                "openab-lab",
                &project.to_string_lossy(),
            )
            .expect("record");
        let binding = store
            .lookup_binding(&ctx("555555555555555555"))
            .expect("lookup")
            .expect("binding");
        assert_eq!(binding.workspace_name, "openab-lab");
        assert_eq!(binding.resolved_cwd, project.to_string_lossy());
        assert_eq!(binding.primary_bot.as_deref(), Some("codex"));
        assert_eq!(binding.primary_bot_discord_user_id.as_deref(), Some("9001"));
    }

    #[test]
    fn repeated_same_workspace_preserves_primary() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let project = root.join("openab-lab");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&project).expect("project");
        let store_a = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store a");
        let store_b =
            ThreadBindingStore::from_config(config(&root, &store_dir, "reviewer"), Some(""))
                .expect("store b");

        store_a
            .record_binding(
                &ctx("555555555555555555"),
                "openab-lab",
                &project.to_string_lossy(),
            )
            .expect("record");
        store_b
            .handover_primary(
                &ctx("555555555555555555"),
                "9002",
                PrimaryUpdateActor::Human {
                    user_id: "333333333333333333".to_string(),
                },
                "777777777777777777",
            )
            .expect("handover");
        store_a
            .record_binding(
                &ctx("555555555555555555"),
                "openab-lab",
                &project.to_string_lossy(),
            )
            .expect("same workspace");

        let binding = store_b
            .lookup_binding(&ctx("555555555555555555"))
            .expect("lookup")
            .expect("binding");
        assert_eq!(binding.primary_bot.as_deref(), Some("reviewer"));
    }

    #[test]
    fn rejects_agent_not_allowed_in_profile() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let project = root.join("openab-lab");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&project).expect("project");
        let store =
            ThreadBindingStore::from_config(config(&root, &store_dir, "outsider"), Some(""))
                .expect("store");
        let err = store
            .record_binding(
                &ctx("555555555555555555"),
                "openab-lab",
                &project.to_string_lossy(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("not allowed"));
    }

    #[test]
    fn rejects_parent_channel_mismatch() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let project = root.join("openab-lab");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&project).expect("project");
        let store = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store");
        let mut bad_ctx = ctx("555555555555555555");
        bad_ctx.parent_channel_id = Some("999999999999999999".to_string());
        let err = store
            .record_binding(&bad_ctx, "openab-lab", &project.to_string_lossy())
            .unwrap_err();
        assert!(err.to_string().contains("no ChannelProfile"));
    }

    #[test]
    fn rejects_cwd_outside_profile_root() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let outside = tmp.path().join("outside");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&root).expect("workspace root");
        fs::create_dir_all(&outside).expect("outside dir");
        let store = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store");

        let err = store
            .record_binding(
                &ctx("555555555555555555"),
                "outside",
                &outside.to_string_lossy(),
            )
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("outside the ChannelProfile workspace root"));
    }

    #[test]
    fn lookup_rejects_deleted_binding_cwd() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let project = root.join("openab-lab");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&project).expect("project");
        let store = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store");

        store
            .record_binding(
                &ctx("555555555555555555"),
                "openab-lab",
                &project.to_string_lossy(),
            )
            .expect("record");
        fs::remove_dir(&project).expect("delete project");

        let err = store
            .lookup_binding(&ctx("555555555555555555"))
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("failed to resolve thread binding cwd"));
    }

    #[test]
    fn concurrent_writes_keep_both_bindings() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let project_a = root.join("a");
        let project_b = root.join("b");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&project_a).expect("project a");
        fs::create_dir_all(&project_b).expect("project b");
        let config_a = config(&root, &store_dir, "codex");
        let config_b = config(&root, &store_dir, "reviewer");
        let root_a = project_a.to_string_lossy().to_string();
        let root_b = project_b.to_string_lossy().to_string();

        let handle_a = std::thread::spawn(move || {
            let store = ThreadBindingStore::from_config(config_a, Some("")).expect("store a");
            store
                .record_binding(&ctx("555555555555555555"), "a", &root_a)
                .expect("record a");
        });
        let handle_b = std::thread::spawn(move || {
            let store = ThreadBindingStore::from_config(config_b, Some("")).expect("store b");
            store
                .record_binding(&ctx("666666666666666666"), "b", &root_b)
                .expect("record b");
        });
        handle_a.join().expect("join a");
        handle_b.join().expect("join b");

        let store = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store");
        assert!(store
            .lookup_binding(&ctx("555555555555555555"))
            .expect("lookup a")
            .is_some());
        assert!(store
            .lookup_binding(&ctx("666666666666666666"))
            .expect("lookup b")
            .is_some());
    }

    #[test]
    fn acquire_lock_reclaims_stale_lock() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&store_dir).expect("store dir");
        let store = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store");

        // Simulate a lock left behind by a writer that crashed mid-update.
        let lock_path = store.lock_path();
        fs::write(&lock_path, b"").expect("seed stale lock");
        assert!(lock_path.exists());

        // A zero staleness window treats the orphaned lock as reclaimable, so
        // the atomic-rename steal succeeds and acquisition proceeds instead of
        // wedging until the 5s retry budget is exhausted.
        let guard = store
            .acquire_lock_with(Duration::from_secs(0))
            .expect("reclaim stale lock");
        drop(guard);
        assert!(!lock_path.exists());
    }

    #[test]
    fn acquire_lock_keeps_fresh_lock_held_by_live_holder() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&store_dir).expect("store dir");
        let store = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store");

        // A live holder's lock is younger than the staleness window and must
        // not be stolen, so a second acquisition times out rather than racing
        // in alongside the first guard.
        let guard = store.acquire_lock().expect("first lock");
        let err = store
            .acquire_lock_with(Duration::from_secs(LOCK_STALE_SECS))
            .unwrap_err();
        assert!(err.to_string().contains("timed out"));
        drop(guard);
    }

    #[test]
    fn detects_profile_agent_mentions() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let project = root.join("openab-lab");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&project).expect("project");
        let store = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store");
        assert!(store
            .message_mentions_profile_agent(&ctx("555555555555555555"), "hello <@9002>")
            .expect("mention"));
        assert!(store
            .message_mentions_profile_agent(&ctx("555555555555555555"), "hello <@!9002>")
            .expect("legacy mention"));
        assert!(store
            .message_mentions_profile_agent(&ctx("555555555555555555"), "hello <@&9902>")
            .expect("role mention"));
        assert!(!store
            .message_mentions_profile_agent(&ctx("555555555555555555"), "hello B")
            .expect("alias"));
    }

    #[test]
    fn human_handover_updates_primary_to_target_agent() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let project = root.join("openab-lab");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&project).expect("project");
        let store_a = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store a");
        let store_b =
            ThreadBindingStore::from_config(config(&root, &store_dir, "reviewer"), Some(""))
                .expect("store b");

        store_a
            .record_binding(
                &ctx("555555555555555555"),
                "openab-lab",
                &project.to_string_lossy(),
            )
            .expect("record");
        let binding = store_b
            .handover_primary(
                &ctx("555555555555555555"),
                "9002",
                PrimaryUpdateActor::Human {
                    user_id: "333333333333333333".to_string(),
                },
                "777777777777777777",
            )
            .expect("handover");

        assert_eq!(binding.primary_bot.as_deref(), Some("reviewer"));
        assert_eq!(binding.primary_bot_discord_user_id.as_deref(), Some("9002"));
        assert_eq!(
            binding.primary_update_message_id.as_deref(),
            Some("777777777777777777")
        );
        assert!(store_b
            .current_agent_is_primary(&ctx("555555555555555555"), "9002")
            .expect("primary"));
        assert!(!store_a
            .current_agent_is_primary(&ctx("555555555555555555"), "9001")
            .expect("not primary"));
    }

    #[test]
    fn current_primary_bot_can_handover_but_non_primary_bot_cannot() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let project = root.join("openab-lab");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&project).expect("project");
        let store_a = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store a");
        let store_b =
            ThreadBindingStore::from_config(config(&root, &store_dir, "reviewer"), Some(""))
                .expect("store b");

        store_a
            .record_binding(
                &ctx("555555555555555555"),
                "openab-lab",
                &project.to_string_lossy(),
            )
            .expect("record");
        store_b
            .handover_primary(
                &ctx("555555555555555555"),
                "9002",
                PrimaryUpdateActor::Bot {
                    user_id: "9001".to_string(),
                },
                "777777777777777777",
            )
            .expect("current primary handover");

        let err = store_a
            .handover_primary(
                &ctx("555555555555555555"),
                "9001",
                PrimaryUpdateActor::Bot {
                    user_id: "9001".to_string(),
                },
                "888888888888888888",
            )
            .unwrap_err();
        assert!(err.to_string().contains("current primary"));
    }

    #[test]
    fn invalid_handover_target_preserves_primary() {
        let tmp = TempDir::new().expect("tmp");
        let root = tmp.path().join("workspace");
        let project = root.join("openab-lab");
        let store_dir = root.join(".openab");
        fs::create_dir_all(&project).expect("project");
        let store = ThreadBindingStore::from_config(config(&root, &store_dir, "codex"), Some(""))
            .expect("store");
        store
            .record_binding(
                &ctx("555555555555555555"),
                "openab-lab",
                &project.to_string_lossy(),
            )
            .expect("record");
        let err = store
            .handover_primary(
                &ctx("555555555555555555"),
                "9999",
                PrimaryUpdateActor::Human {
                    user_id: "333333333333333333".to_string(),
                },
                "777777777777777777",
            )
            .unwrap_err();
        assert!(err.to_string().contains("not in ChannelProfile"));
        let binding = store
            .lookup_binding(&ctx("555555555555555555"))
            .expect("lookup")
            .expect("binding");
        assert_eq!(binding.primary_bot.as_deref(), Some("codex"));
    }
}
