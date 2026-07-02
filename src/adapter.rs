use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tracing::{error, warn};

use crate::acp::{classify_notification, parse_turn_result, AcpEvent, ContentBlock, SessionPool};
use crate::config::{AgentAttachmentsConfig, ReactionsConfig, ToolDisplay, WorkspaceRequest};
use crate::error_display::{format_coded_error, format_user_error};
use crate::format;
use crate::markdown::{self, TableMode};
use crate::reactions::StatusReactionController;

// --- Output directive parsing ---

/// Parsed directives from agent output.
#[derive(Default, Debug)]
pub struct OutputDirectives {
    /// Message ID to reply to (Discord: message_reference)
    pub reply_to: Option<String>,
    /// File paths requested for upload into the outbound platform message.
    pub attachments: Vec<String>,
}

fn valid_reply_to_value(v: &str) -> bool {
    !v.is_empty()
        && v.len() <= 64
        && v.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_')
}

fn parse_output_directive_inner(inner: &str, directives: &mut OutputDirectives) -> bool {
    let Some((key, value)) = inner.split_once(':') else {
        return false;
    };

    match key.trim() {
        "reply_to" => {
            let v = value.trim();
            if valid_reply_to_value(v) {
                directives.reply_to = Some(v.to_string());
            }
        }
        "attach" => {
            let v = value.trim();
            if !v.is_empty() {
                directives.attachments.push(v.to_string());
            }
        }
        _ => {
            tracing::debug!(key = key.trim(), "unknown output directive ignored");
        }
    }

    true
}

fn late_attach_has_safe_boundary(after_close: &str) -> bool {
    after_close.is_empty() || after_close.starts_with('\n') || after_close.starts_with("\r\n")
}

fn parse_late_output_directive(
    inner: &str,
    after_close: &str,
    directives: &mut OutputDirectives,
) -> bool {
    let Some((key, value)) = inner.split_once(':') else {
        return false;
    };

    let key = key.trim();
    let v = value.trim();
    match key {
        "reply_to" if valid_reply_to_value(v) => {
            directives.reply_to = Some(v.to_string());
            true
        }
        "attach" if !v.is_empty() && late_attach_has_safe_boundary(after_close) => {
            directives.attachments.push(v.to_string());
            true
        }
        _ => false,
    }
}

fn strip_late_output_directives(content: &str, directives: &mut OutputDirectives) -> String {
    let mut output = String::with_capacity(content.len());
    let mut rest = content;

    while let Some(open_pos) = rest.find("[[") {
        output.push_str(&rest[..open_pos]);
        let candidate = &rest[open_pos + 2..];
        let Some(close_pos) = candidate.find("]]") else {
            output.push_str(&rest[open_pos..]);
            return output;
        };

        let inner = &candidate[..close_pos];
        let after_close = &candidate[close_pos + 2..];

        if parse_late_output_directive(inner, after_close, directives) {
            rest = after_close;
            if output.ends_with('\n') && rest.starts_with('\n') {
                rest = &rest[1..];
            }
            if let Some(stripped) = rest.strip_prefix(' ') {
                rest = stripped;
            }
            let before = output.chars().next_back();
            let after = rest.chars().next();
            if before.is_some_and(|c| !c.is_whitespace())
                && after.is_some_and(|c| !c.is_whitespace())
            {
                output.push(' ');
            }
        } else {
            output.push_str(&rest[open_pos..open_pos + 2 + close_pos + 2]);
            rest = &candidate[close_pos + 2..];
        }
    }

    output.push_str(rest);
    output
}

/// Parse `[[key:value]]` directives from agent output.
///
/// The preferred form is a directive header at the beginning of output. For
/// `reply_to` and newline-terminated `attach`, also accept a late directive
/// inside final output because real agents often preface tool-mediated
/// operations with a short explanation before the directive. This still runs
/// only after the agent turn finishes, before final message creation.
/// Returns parsed directives and the remaining content (directives stripped).
pub fn parse_output_directives(content: &str) -> (OutputDirectives, String) {
    let mut directives = OutputDirectives::default();
    let mut content_start = 0usize;
    let mut consumed_directive = false;

    loop {
        let rest = &content[content_start..];
        let Some(after_open) = rest.strip_prefix("[[") else {
            break;
        };
        let Some(close_pos) = after_open.find("]]") else {
            break;
        };
        let inner = &after_open[..close_pos];
        if !parse_output_directive_inner(inner, &mut directives) {
            break;
        }

        consumed_directive = true;
        content_start += 2 + close_pos + 2;

        while content_start < content.len() {
            let rest = &content[content_start..];
            if let Some(stripped) = rest.strip_prefix("\r\n") {
                content_start = content.len() - stripped.len();
            } else if let Some(stripped) = rest.strip_prefix('\n') {
                content_start = content.len() - stripped.len();
            } else if rest.starts_with(' ') || rest.starts_with('\t') {
                content_start += 1;
            } else {
                break;
            }

            if content[content_start..].starts_with("[[") {
                break;
            }
            if !matches!(
                content[content_start..].chars().next(),
                Some(' ' | '\t' | '\n' | '\r')
            ) {
                break;
            }
        }

        if !content[content_start..].starts_with("[[") {
            break;
        }
    }

    let remaining = if consumed_directive {
        content[content_start..]
            .trim_start_matches([' ', '\t'])
            .to_string()
    } else {
        content[content_start..].to_string()
    };
    let remaining = strip_late_output_directives(&remaining, &mut directives);
    (directives, remaining)
}

fn attachment_warning(message: impl std::fmt::Display) -> String {
    format!("⚠️ attachment skipped: {message}")
}

fn requested_attachment_path(raw_path: &str, base_dir: &Path) -> PathBuf {
    let path = PathBuf::from(raw_path.trim());
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

fn sanitize_attachment_filename(path: &Path, fallback_idx: usize) -> String {
    let raw = path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("attachment-{fallback_idx}"));

    let mut out = String::with_capacity(raw.len().min(128));
    for ch in raw.chars().take(128) {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() || out == "." || out == ".." {
        format!("attachment-{fallback_idx}")
    } else {
        out
    }
}

pub(crate) fn prepare_output_attachments(
    requested_paths: &[String],
    config: &AgentAttachmentsConfig,
    base_dir: &Path,
) -> (Vec<OutboundAttachment>, Vec<String>) {
    if requested_paths.is_empty() {
        return (Vec::new(), Vec::new());
    }

    if !config.enabled {
        return (
            Vec::new(),
            vec![format!(
                "⚠️ attachment upload is disabled for this bot; {} requested file(s) were not sent.",
                requested_paths.len()
            )],
        );
    }

    if requested_paths.len() > config.max_files {
        return (
            Vec::new(),
            vec![attachment_warning(format!(
                "too many files requested ({} > {})",
                requested_paths.len(),
                config.max_files
            ))],
        );
    }

    let allowed_roots: Vec<PathBuf> = config
        .allowed_paths
        .iter()
        .filter_map(|root| match fs::canonicalize(root) {
            Ok(path) => Some(path),
            Err(e) => {
                tracing::warn!(path = root, error = %e, "attachment allowlist root is unavailable");
                None
            }
        })
        .collect();

    if allowed_roots.is_empty() {
        return (
            Vec::new(),
            vec![attachment_warning(
                "no valid attachment allowlist roots are configured",
            )],
        );
    }

    let mut attachments = Vec::new();
    let mut warnings = Vec::new();
    let mut total_bytes = 0u64;

    for (idx, raw_path) in requested_paths.iter().enumerate() {
        let display_path = raw_path.trim();
        if display_path.is_empty() {
            warnings.push(attachment_warning("empty path"));
            continue;
        }

        let requested_path = requested_attachment_path(display_path, base_dir);
        let canonical = match fs::canonicalize(&requested_path) {
            Ok(path) => path,
            Err(e) => {
                warnings.push(attachment_warning(format!(
                    "{} is missing or cannot be resolved ({e})",
                    display_path
                )));
                continue;
            }
        };

        if !allowed_roots.iter().any(|root| canonical.starts_with(root)) {
            warnings.push(attachment_warning(format!(
                "{} is outside the configured allowlist",
                display_path
            )));
            continue;
        }

        let metadata = match fs::metadata(&canonical) {
            Ok(metadata) => metadata,
            Err(e) => {
                warnings.push(attachment_warning(format!(
                    "{} cannot be inspected ({e})",
                    display_path
                )));
                continue;
            }
        };

        if !metadata.is_file() {
            warnings.push(attachment_warning(format!(
                "{display_path} is not a regular file"
            )));
            continue;
        }

        if metadata.len() > config.max_file_bytes {
            warnings.push(attachment_warning(format!(
                "{} is too large ({} > {} bytes)",
                display_path,
                metadata.len(),
                config.max_file_bytes
            )));
            continue;
        }

        let bytes = match fs::read(&canonical) {
            Ok(bytes) => bytes,
            Err(e) => {
                warnings.push(attachment_warning(format!(
                    "{} cannot be read ({e})",
                    display_path
                )));
                continue;
            }
        };

        total_bytes = total_bytes.saturating_add(bytes.len() as u64);
        attachments.push(OutboundAttachment {
            filename: sanitize_attachment_filename(&requested_path, idx + 1),
            bytes,
        });
    }

    if total_bytes > config.max_total_bytes {
        warnings.push(attachment_warning(format!(
            "total attachment size is too large ({} > {} bytes)",
            total_bytes, config.max_total_bytes
        )));
        attachments.clear();
    }

    (attachments, warnings)
}

fn append_attachment_warnings(content: String, warnings: &[String]) -> String {
    if warnings.is_empty() {
        return content;
    }
    let warning_block = warnings.join("\n");
    if content.trim().is_empty() {
        warning_block
    } else {
        format!("{content}\n\n{warning_block}")
    }
}

// --- Platform-agnostic types ---

/// Identifies a channel or thread across platforms.
///
/// Used for **routing**: `channel_id` is the ID the adapter sends messages to.
/// For Discord threads, this is the thread's own channel ID (Discord API
/// requires it for `say`/`edit`). Use `parent_id` to find the parent channel.
///
/// Compare with `SenderContext`, which is **metadata for the agent**: there
/// `channel_id` is the parent channel and `thread_id` is the thread,
/// matching Slack's model for cross-platform consistency.
#[derive(Clone, Debug)]
pub struct ChannelRef {
    pub platform: String,
    pub channel_id: String,
    /// Thread within a channel (e.g. Slack thread_ts, Telegram topic_id).
    /// For Discord, threads are separate channels so this is None.
    pub thread_id: Option<String>,
    /// Parent channel if this is a thread-as-channel (Discord).
    pub parent_id: Option<String>,
    /// Originating gateway event ID, propagated back in `GatewayReply.reply_to`
    /// so the gateway can correlate replies with inbound events (e.g. LINE reply tokens).
    /// Excluded from Hash/Eq — two ChannelRefs pointing to the same channel are
    /// equal regardless of which event they originated from.
    pub origin_event_id: Option<String>,
}

impl PartialEq for ChannelRef {
    fn eq(&self, other: &Self) -> bool {
        self.platform == other.platform
            && self.channel_id == other.channel_id
            && self.thread_id == other.thread_id
            && self.parent_id == other.parent_id
    }
}

impl Eq for ChannelRef {}

impl std::hash::Hash for ChannelRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.platform.hash(state);
        self.channel_id.hash(state);
        self.thread_id.hash(state);
        self.parent_id.hash(state);
    }
}

/// Identifies a message across platforms.
#[derive(Clone, Debug)]
pub struct MessageRef {
    pub channel: ChannelRef,
    pub message_id: String,
}

/// A broker-validated outbound file upload.
#[derive(Clone, Debug)]
pub struct OutboundAttachment {
    pub filename: String,
    pub bytes: Vec<u8>,
}

/// Bundles per-message parameters for `AdapterRouter::handle_message`.
///
/// Introduced to reduce parameter count and make the signature extensible
/// (e.g. streaming policy, rate limit hints) without breaking call sites.
pub struct MessageContext {
    pub thread_channel: ChannelRef,
    pub sender_json: String,
    pub prompt: String,
    pub session_directives: SessionDirectives,
    pub extra_blocks: Vec<ContentBlock>,
    pub trigger_msg: MessageRef,
    pub other_bot_present: bool,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct SessionDirectives {
    pub workspace: Option<WorkspaceRequest>,
    pub title: Option<String>,
}

impl SessionDirectives {
    pub fn has_any(&self) -> bool {
        self.workspace.is_some() || self.title.is_some()
    }
}

/// Sender identity injected into prompts for downstream agent context.
///
/// This is **metadata for the agent** — `channel_id` always refers to the
/// logical parent channel, and `thread_id` identifies the thread (if any).
/// This convention is consistent across platforms (Slack, Discord, Telegram).
///
/// Compare with `ChannelRef`, which is used for **routing**: there
/// `channel_id` is the ID the adapter sends messages to (for Discord
/// threads, that's the thread's own channel ID, not the parent).
#[derive(Clone, Debug, Serialize)]
pub struct SenderContext {
    pub schema: String,
    pub sender_id: String,
    pub sender_name: String,
    pub display_name: String,
    pub channel: String,
    pub channel_id: String,
    /// Thread identifier, if the message is inside a thread.
    /// Slack: thread_ts. Discord: thread channel ID (channel_id holds the parent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    pub is_bot: bool,
    /// Platform message creation time (ISO 8601 UTC), if available.
    /// Discord/Slack: platform timestamp. Gateway: broker receive time (best-effort).
    /// Additive optional field — schema version stays openab.sender.v1 (no consumer
    /// breakage). If future additions require breaking changes, bump to v1.1+.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Platform message ID. Agents can use this to reply to a specific message
    /// via the `[[reply_to:<message_id>]]` output directive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// The platform user ID of the receiving bot/agent.
    /// Enables agents to identify themselves when multiple agents share the same backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receiver_id: Option<String>,
}

// --- ChatAdapter trait ---

#[async_trait]
pub trait ChatAdapter: Send + Sync + 'static {
    /// Platform name for logging and session key namespacing.
    fn platform(&self) -> &'static str;

    /// Maximum message length for this platform (e.g. 2000 for Discord, 4000 for Slack).
    fn message_limit(&self) -> usize;

    /// Send a new message, returns a reference to the sent message.
    async fn send_message(&self, channel: &ChannelRef, content: &str) -> Result<MessageRef>;

    /// Create a thread from a trigger message, returns the thread channel ref.
    async fn create_thread(
        &self,
        channel: &ChannelRef,
        trigger_msg: &MessageRef,
        title: &str,
    ) -> Result<ChannelRef>;

    /// Add a reaction/emoji to a message.
    async fn add_reaction(&self, msg: &MessageRef, emoji: &str) -> Result<()>;

    /// Remove a reaction/emoji from a message.
    async fn remove_reaction(&self, msg: &MessageRef, emoji: &str) -> Result<()>;

    /// Edit an existing message in-place (for streaming updates).
    /// Default: unsupported (send-once only).
    async fn edit_message(&self, _msg: &MessageRef, _content: &str) -> Result<()> {
        Err(anyhow::anyhow!("edit_message not supported"))
    }

    /// Send a message as a reply to a specific message (Discord: message_reference).
    /// Default: falls back to plain send_message (ignores reply_to).
    async fn send_message_with_reply(
        &self,
        channel: &ChannelRef,
        content: &str,
        reply_to_message_id: &str,
    ) -> Result<MessageRef> {
        let _ = reply_to_message_id; // unused in default impl
        self.send_message(channel, content).await
    }

    /// Send a message with already validated file attachments.
    ///
    /// Default: non-attachment adapters strip the directives and return a
    /// visible warning instead of trying to expose local file paths.
    async fn send_message_with_attachments(
        &self,
        channel: &ChannelRef,
        content: &str,
        attachments: &[OutboundAttachment],
        reply_to_message_id: Option<&str>,
    ) -> Result<MessageRef> {
        let warning = format!(
            "⚠️ attachment upload is not supported on {} for {} requested file(s).",
            self.platform(),
            attachments.len()
        );
        let content = if content.trim().is_empty() {
            warning
        } else {
            format!("{content}\n\n{warning}")
        };
        if let Some(reply_to_message_id) = reply_to_message_id {
            self.send_message_with_reply(channel, &content, reply_to_message_id)
                .await
        } else {
            self.send_message(channel, &content).await
        }
    }

    /// Delete a message. Used to remove streaming placeholders when reply_to is set.
    /// Default: edits to zero-width space (fallback for platforms without delete support).
    async fn delete_message(&self, msg: &MessageRef) -> Result<()> {
        self.edit_message(msg, "\u{200b}").await
    }

    /// Whether this adapter should use streaming edit (true) or send-once (false).
    /// `other_bot_present` indicates if another bot has posted in the current thread.
    /// Streaming should be disabled in multi-bot threads to avoid edit interference.
    /// NOTE: Slight race window exists — the multibot cache is checked before
    /// handle_message, so a bot arriving between the check and the response will
    /// not be detected until the next message. This is acceptable: the first
    /// response may stream, but subsequent ones will correctly use send-once.
    fn use_streaming(&self, other_bot_present: bool) -> bool;
}

// --- AdapterRouter ---

/// Shared logic for routing messages to ACP agents, managing sessions,
/// streaming edits, and controlling reactions. Platform-independent.
pub struct AdapterRouter {
    pool: Arc<SessionPool>,
    reactions_config: ReactionsConfig,
    attachments_config: AgentAttachmentsConfig,
    table_mode: TableMode,
    prompt_hard_timeout: std::time::Duration,
    /// Polling cadence for the recv-loop liveness check (#732).
    liveness_check_interval: std::time::Duration,
}

impl AdapterRouter {
    pub fn new(
        pool: Arc<SessionPool>,
        reactions_config: ReactionsConfig,
        attachments_config: AgentAttachmentsConfig,
        table_mode: TableMode,
        prompt_hard_timeout_secs: u64,
        liveness_check_secs: u64,
    ) -> Self {
        if liveness_check_secs >= prompt_hard_timeout_secs {
            warn!(
                liveness_check_secs,
                prompt_hard_timeout_secs,
                "pool.liveness_check_secs >= pool.prompt_hard_timeout_secs; \
                 the hard ceiling will only fire after the next liveness tick \
                 and may be effectively bypassed. Lower liveness_check_secs."
            );
        }
        Self {
            pool,
            reactions_config,
            attachments_config,
            table_mode,
            prompt_hard_timeout: std::time::Duration::from_secs(prompt_hard_timeout_secs),
            liveness_check_interval: std::time::Duration::from_secs(liveness_check_secs),
        }
    }

    /// Access the underlying session pool (e.g. for config option queries).
    pub fn pool(&self) -> &Arc<SessionPool> {
        &self.pool
    }

    /// Access the reactions config (used by dispatch.rs).
    pub fn reactions_config(&self) -> &ReactionsConfig {
        &self.reactions_config
    }

    /// Pack one arrival event into ContentBlocks. Per-arrival layout:
    ///   Text { "<sender_context>\n{json}\n</sender_context>" }   <- delimiter
    ///   [Text blocks from extra_blocks (e.g. STT transcripts)]
    ///   Text { "{prompt}" }                                       <- omitted if empty
    ///   [non-Text blocks from extra_blocks (e.g. Image)]
    ///
    /// The sender_context block stands alone so it can serve as a structural
    /// delimiter between arrivals in batched dispatch — agents can scan for
    /// `<sender_context>` openers to find arrival boundaries. Within an arrival,
    /// transcript text precedes the typed prompt to match pre-batching adapter
    /// behavior (voice content first), and images trail the prompt as before.
    /// This is the single packing code path for both per-message and batched
    /// dispatch (ADR §3.5). For a batch of N messages, call this N times and
    /// concatenate.
    pub fn pack_arrival_event(
        sender_json: &str,
        prompt: &str,
        extra_blocks: Vec<ContentBlock>,
    ) -> Vec<ContentBlock> {
        let header = format!("<sender_context>\n{}\n</sender_context>", sender_json);
        let (texts, others): (Vec<_>, Vec<_>) = extra_blocks
            .into_iter()
            .partition(|b| matches!(b, ContentBlock::Text { .. }));
        let mut blocks = Vec::with_capacity(2 + texts.len() + others.len());
        blocks.push(ContentBlock::Text { text: header });
        blocks.extend(texts);
        if !prompt.is_empty() {
            blocks.push(ContentBlock::Text {
                text: prompt.to_string(),
            });
        }
        blocks.extend(others);
        blocks
    }

    /// Parse user-facing session initialization directives.
    /// Supported input directives are `[[ws:name]]`, `[[ws:name --create]]`,
    /// and `[[title:...]]`. Directives may be inline or on separate lines and
    /// are stripped before the prompt reaches the agent. Unknown keys are
    /// treated as errors so typos do not silently change behavior.
    pub fn parse_session_directives(prompt: &str) -> Result<(SessionDirectives, String)> {
        let mut directives = SessionDirectives::default();
        let mut cleaned = String::with_capacity(prompt.len());
        let mut rest = prompt;

        while let Some(open_idx) = rest.find("[[") {
            cleaned.push_str(&rest[..open_idx]);
            let after_open = &rest[open_idx + 2..];
            let Some(close_idx) = after_open.find("]]") else {
                return Err(anyhow!("unterminated session directive"));
            };
            let inner = after_open[..close_idx].trim();
            let Some((raw_key, raw_value)) = inner.split_once(':') else {
                return Err(anyhow!("malformed session directive: [[{inner}]]"));
            };
            let key = raw_key.trim();
            let value = raw_value.trim();
            match key {
                "ws" => {
                    directives.workspace = Some(Self::parse_workspace_directive(value)?);
                }
                "title" => {
                    if value.is_empty() {
                        return Err(anyhow!("empty title directive"));
                    }
                    directives.title = Some(value.to_string());
                }
                "" => return Err(anyhow!("empty session directive key")),
                other => return Err(anyhow!("unknown session directive: {other}")),
            }
            rest = &after_open[close_idx + 2..];
        }
        cleaned.push_str(rest);

        Ok((directives, Self::clean_stripped_prompt(&cleaned)))
    }

    fn parse_workspace_directive(value: &str) -> Result<WorkspaceRequest> {
        let mut parts = value.split_whitespace();
        let Some(name) = parts.next() else {
            return Err(anyhow!("empty workspace directive"));
        };
        let mut create = false;
        for flag in parts {
            match flag {
                "--create" => create = true,
                other => return Err(anyhow!("unknown workspace flag: {other}")),
            }
        }
        if create {
            Ok(WorkspaceRequest::Create(name.to_string()))
        } else {
            Ok(WorkspaceRequest::Existing(name.to_string()))
        }
    }

    fn clean_stripped_prompt(prompt: &str) -> String {
        let lines: Vec<&str> = prompt.lines().collect();
        let first = lines.iter().position(|line| !line.trim().is_empty());
        let last = lines.iter().rposition(|line| !line.trim().is_empty());
        let Some(first) = first else {
            return String::new();
        };
        let Some(last) = last else {
            return String::new();
        };
        lines[first..=last]
            .iter()
            .map(|line| line.trim())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Handle an incoming user message. The adapter is responsible for
    /// filtering, resolving the thread, and building the SenderContext.
    /// This method handles sender context injection, session management, and streaming.
    pub async fn handle_message(
        &self,
        adapter: &Arc<dyn ChatAdapter>,
        ctx: MessageContext,
    ) -> Result<()> {
        tracing::debug!(platform = adapter.platform(), "processing message");

        let (parsed_directives, prompt) = Self::parse_session_directives(&ctx.prompt)?;
        let session_directives = if ctx.session_directives.has_any() {
            ctx.session_directives
        } else {
            parsed_directives
        };
        let content_blocks = Self::pack_arrival_event(&ctx.sender_json, &prompt, ctx.extra_blocks);

        let thread_key = format!(
            "{}:{}",
            adapter.platform(),
            ctx.thread_channel
                .thread_id
                .as_deref()
                .unwrap_or(&ctx.thread_channel.channel_id)
        );

        if let Err(e) = self
            .pool
            .get_or_create(&thread_key, session_directives.workspace.as_ref(), None)
            .await
        {
            let msg = format_user_error(&e.to_string());
            let _ = adapter
                .send_message(&ctx.thread_channel, &format!("⚠️ {msg}"))
                .await;
            error!("pool error: {e}");
            return Err(e);
        }

        let reactions = Arc::new(StatusReactionController::new(
            self.reactions_config.enabled,
            adapter.clone(),
            vec![ctx.trigger_msg.clone()],
            self.reactions_config.emojis.clone(),
            self.reactions_config.timing.clone(),
        ));
        reactions.set_queued().await;

        let result = self
            .stream_prompt(
                adapter,
                &thread_key,
                content_blocks,
                &ctx.thread_channel,
                reactions.clone(),
                ctx.other_bot_present,
            )
            .await;

        match &result {
            Ok(()) => reactions.set_done().await,
            Err(_) => reactions.set_error().await,
        }

        let hold_ms = if result.is_ok() {
            self.reactions_config.timing.done_hold_ms
        } else {
            self.reactions_config.timing.error_hold_ms
        };
        if self.reactions_config.remove_after_reply {
            let reactions = reactions;
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(hold_ms)).await;
                reactions.clear().await;
            });
        }

        if let Err(ref e) = result {
            let _ = adapter
                .send_message(&ctx.thread_channel, &format!("⚠️ {e}"))
                .await;
        }

        result
    }

    async fn stream_prompt(
        &self,
        adapter: &Arc<dyn ChatAdapter>,
        thread_key: &str,
        content_blocks: Vec<ContentBlock>,
        thread_channel: &ChannelRef,
        reactions: Arc<StatusReactionController>,
        other_bot_present: bool,
    ) -> Result<()> {
        self.stream_prompt_blocks(
            adapter,
            thread_key,
            content_blocks,
            thread_channel,
            reactions,
            other_bot_present,
        )
        .await
    }

    /// Drive one ACP turn with the given pre-packed ContentBlocks.
    /// Called by both `handle_message` (per-message mode) and `dispatch::dispatch_batch`
    /// (batched mode).
    pub async fn stream_prompt_blocks(
        &self,
        adapter: &Arc<dyn ChatAdapter>,
        thread_key: &str,
        content_blocks: Vec<ContentBlock>,
        thread_channel: &ChannelRef,
        reactions: Arc<StatusReactionController>,
        other_bot_present: bool,
    ) -> Result<()> {
        let adapter = adapter.clone();
        let thread_channel = thread_channel.clone();
        let message_limit = adapter.message_limit();
        let streaming = adapter.use_streaming(other_bot_present);
        let table_mode = self.table_mode;
        let tool_display = self.reactions_config.tool_display;
        let attachments_config = self.attachments_config.clone();
        let attachment_base_dir = match self.pool.working_dir_for_thread(thread_key).await {
            Ok(dir) => PathBuf::from(dir),
            Err(e) => {
                warn!(thread_key, error = %e, "failed to resolve attachment base directory");
                PathBuf::from("/tmp")
            }
        };
        let prompt_hard_timeout = self.prompt_hard_timeout;
        let liveness_check_interval = self.liveness_check_interval;

        self.pool
            .with_connection(thread_key, |conn| {
                let content_blocks = content_blocks.clone();
                Box::pin(async move {
                    let reset = conn.session_reset;
                    conn.session_reset = false;

                    let (mut rx, request_id) = conn.session_prompt(content_blocks).await?;
                    reactions.set_thinking().await;

                    let mut text_buf = String::new();
                    let mut tool_lines: Vec<ToolEntry> = Vec::new();

                    if reset {
                        text_buf.push_str("⚠️ _Session expired, starting fresh..._\n\n");
                    }

                    // Streaming edit: send placeholder, spawn edit loop
                    let (buf_tx, placeholder_msg) = if streaming {
                        let initial = if reset {
                            "⚠️ _Session expired, starting fresh..._\n\n…".to_string()
                        } else {
                            "…".to_string()
                        };
                        let msg = adapter.send_message(&thread_channel, &initial).await?;
                        let (tx, rx) = tokio::sync::watch::channel(initial);
                        let edit_adapter = adapter.clone();
                        let edit_msg = msg.clone();
                        let limit = message_limit;
                        let mut buf_rx = rx;
                        tokio::spawn(async move {
                            let mut last = String::new();
                            loop {
                                tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                                if buf_rx.has_changed().unwrap_or(false) {
                                    let content = buf_rx.borrow_and_update().clone();
                                    if content != last {
                                        let display = if content.chars().count() > limit - 100 {
                                            format!(
                                                "…{}",
                                                format::truncate_chars_tail(&content, limit - 100)
                                            )
                                        } else {
                                            content.clone()
                                        };
                                        let _ =
                                            edit_adapter.edit_message(&edit_msg, &display).await;
                                        last = content;
                                    }
                                }
                                if buf_rx.has_changed().is_err() {
                                    break;
                                }
                            }
                        });
                        (Some(tx), Some(msg))
                    } else {
                        (None, None)
                    };

                    // (#732) Liveness-aware recv loop. Filters stale id-bearing
                    // messages and abandons cleanly on dead agent / hard ceiling
                    // so late responses cannot leak into the next prompt.
                    let mut response_error: Option<String> = None;
                    // (bd16546) Soft diagnostic for a clean end_turn with 0 output tokens.
                    // Kept separate from response_error because it must surface ONLY when
                    // the reply would otherwise be "_(no response)_" — a turn that streamed
                    // text or tool activity is not silent, and prefixing it with a scary
                    // provider-failure warning would be a false positive (e.g. a bridge
                    // that stubs usage to zeros).
                    let mut silent_turn_diag: Option<String> = None;
                    let prompt_start = tokio::time::Instant::now();
                    loop {
                        let notification = tokio::select! {
                            msg = rx.recv() => match msg {
                                Some(n) => n,
                                // (#1198) Reader saw EOF: the agent's stdout closed. A
                                // successful turn is always signalled by the id-bearing
                                // response to session/prompt, which breaks the loop at the id
                                // branch below *before* any EOF — so reaching this arm means
                                // the turn ended without a final response, i.e. the agent
                                // terminated abnormally (a bridged agent that crashes on a
                                // backend error such as HTTP 500 / quota exhaustion exits
                                // without ever emitting an ACP error notification). Surface it
                                // explicitly instead of silently falling through to
                                // "_(no response)_" or presenting a partial buffer as complete.
                                None => {
                                    if response_error.is_none() {
                                        response_error =
                                            Some("Agent process exited unexpectedly".into());
                                    }
                                    break;
                                }
                            },
                            _ = tokio::time::sleep(liveness_check_interval) => {
                                if !conn.alive() {
                                    response_error = Some("Agent process died".into());
                                    conn.abandon_request(request_id).await;
                                    break;
                                }
                                if prompt_start.elapsed() > prompt_hard_timeout {
                                    response_error = Some(format!(
                                        "Agent exceeded hard timeout ({}s)",
                                        prompt_hard_timeout.as_secs(),
                                    ));
                                    conn.abandon_request(request_id).await;
                                    break;
                                }
                                continue;
                            }
                        };
                        if let Some(notification_id) = notification.id {
                            if notification_id != request_id {
                                // Stale response from a previously-abandoned prompt.
                                // No automated test seam: this path only triggers when a
                                // real subprocess emits a late response after the broker
                                // already called abandon_request — covered by manual
                                // repro against a live agent (see #732 PR description).
                                continue;
                            }
                            if let Some(ref err) = notification.error {
                                response_error = Some(format_coded_error(err.code, &err.message, err.data_message()));
                            } else if let Some(ref result) = notification.result {
                                // (bd16546) A turn that ends with stopReason="end_turn" but
                                // zero output tokens is a strong signal of a silent
                                // provider/auth failure. Record it as a SOFT diagnostic: it
                                // replaces "_(no response)_" below, but never prefixes a
                                // reply that actually produced text/tool output.
                                if parse_turn_result(result).is_silent_failure() {
                                    tracing::warn!(
                                        "turn ended with end_turn and 0 output tokens \
                                         (possible silent provider/auth failure)"
                                    );
                                    silent_turn_diag = Some(
                                        "Agent ended the turn with no output (0 output tokens) \
                                         — likely a silent provider or auth failure"
                                            .into(),
                                    );
                                }
                            }
                            break;
                        }

                        if let Some(event) = classify_notification(&notification) {
                            match event {
                                AcpEvent::Text(t) => {
                                    text_buf.push_str(&t);
                                    if let Some(tx) = &buf_tx {
                                        let _ = tx.send(compose_display(
                                            &tool_lines,
                                            &text_buf,
                                            true,
                                            tool_display,
                                        ));
                                    }
                                }
                                AcpEvent::Thinking => {
                                    reactions.set_thinking().await;
                                }
                                AcpEvent::ToolStart { id, title } if !title.is_empty() => {
                                    reactions.set_tool(&title).await;
                                    let title = sanitize_title(&title);
                                    if let Some(slot) = tool_lines.iter_mut().find(|e| e.id == id) {
                                        slot.title = title;
                                        slot.state = ToolState::Running;
                                    } else {
                                        tool_lines.push(ToolEntry {
                                            id,
                                            title,
                                            state: ToolState::Running,
                                        });
                                    }
                                    if let Some(tx) = &buf_tx {
                                        let _ = tx.send(compose_display(
                                            &tool_lines,
                                            &text_buf,
                                            true,
                                            tool_display,
                                        ));
                                    }
                                }
                                AcpEvent::ToolDone { id, title, status } => {
                                    reactions.set_thinking().await;
                                    let new_state = if status == "completed" {
                                        ToolState::Completed
                                    } else {
                                        ToolState::Failed
                                    };
                                    if let Some(slot) = tool_lines.iter_mut().find(|e| e.id == id) {
                                        if !title.is_empty() {
                                            slot.title = sanitize_title(&title);
                                        }
                                        slot.state = new_state;
                                    } else if !title.is_empty() {
                                        tool_lines.push(ToolEntry {
                                            id,
                                            title: sanitize_title(&title),
                                            state: new_state,
                                        });
                                    }
                                    if let Some(tx) = &buf_tx {
                                        let _ = tx.send(compose_display(
                                            &tool_lines,
                                            &text_buf,
                                            true,
                                            tool_display,
                                        ));
                                    }
                                }
                                AcpEvent::ConfigUpdate { options } => {
                                    conn.config_options = options;
                                }
                                _ => {}
                            }
                        }
                    }

                    conn.prompt_done().await;
                    // Stop the edit loop
                    drop(buf_tx);

                    // Parse output directives from raw text_buf BEFORE compose_display.
                    // Directives are agent meta-layer, not content — must be stripped
                    // before tool lines are composed into the display output.
                    let (directives, stripped_text) = parse_output_directives(&text_buf);
                    let text_buf = stripped_text;
                    let (attachments, attachment_warnings) = prepare_output_attachments(
                        &directives.attachments,
                        &attachments_config,
                        &attachment_base_dir,
                    );

                    // Build final content
                    let final_content =
                        compose_display(&tool_lines, &text_buf, false, tool_display);
                    let final_content = if final_content.is_empty() {
                        if let Some(err) = response_error.as_ref() {
                            format!("⚠️ {err}")
                        } else if !attachments.is_empty() {
                            "Attached requested file(s).".to_string()
                        } else if !attachment_warnings.is_empty() {
                            String::new()
                        } else if let Some(diag) = silent_turn_diag.as_ref() {
                            // (bd16546) exactly the case that would otherwise render as
                            // "_(no response)_" — surface the diagnostic instead.
                            format!("⚠️ {diag}")
                        } else {
                            "_(no response)_".to_string()
                        }
                    } else if let Some(err) = response_error.as_ref() {
                        format!("⚠️ {err}\n\n{final_content}")
                    } else {
                        final_content
                    };

                    let final_content = append_attachment_warnings(final_content, &attachment_warnings);
                    let final_content = markdown::convert_tables(&final_content, table_mode);
                    // (#1153) On Discord a reply that exceeds the 2000-char limit is split
                    // into multiple messages, but only the chunk carrying the original
                    // @mention passes a receiving bot's mention gate (allowBotMessages
                    // = "mentions" / trustedBotIds-only-when-mentioned). Subsequent chunks
                    // are rejected and their content is silently lost. Propagate every
                    // mention to all chunks so the whole handoff lands in one ACP turn.
                    // Pre-deduct the mention footer from the split limit so an appended
                    // footer never busts the hard message-length ceiling.
                    let chunks = if adapter.platform() == "discord" {
                        let mentions = extract_mentions(&final_content);
                        let mention_reserve = mention_footer_len(&mentions);
                        let chunks = format::split_message(
                            &final_content,
                            message_limit.saturating_sub(mention_reserve),
                        );
                        propagate_mentions_to_chunks(chunks, &mentions, message_limit)
                    } else {
                        format::split_message(&final_content, message_limit)
                    };
                    if !attachments.is_empty() {
                        let first_chunk = chunks
                            .first()
                            .map(String::as_str)
                            .unwrap_or("Attached requested file(s).");
                        match adapter
                            .send_message_with_attachments(
                                &thread_channel,
                                first_chunk,
                                &attachments,
                                directives.reply_to.as_deref(),
                            )
                            .await
                        {
                            Ok(_) => {
                                if let Some(msg) = placeholder_msg {
                                    if let Err(e) = adapter.delete_message(&msg).await {
                                        tracing::warn!(error = ?e, "delete placeholder failed; placeholder will remain visible");
                                    }
                                }
                                for chunk in chunks.iter().skip(1) {
                                    let _ = adapter.send_message(&thread_channel, chunk).await;
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = ?e, "attachment send failed");
                                let warning = format!("⚠️ attachment send failed: {e}");
                                if let Some(msg) = placeholder_msg {
                                    let _ = adapter.edit_message(&msg, &warning).await;
                                } else {
                                    let _ = adapter.send_message(&thread_channel, &warning).await;
                                }
                            }
                        }
                    } else if let Some(msg) = placeholder_msg {
                        if let Some(ref reply_id) = directives.reply_to {
                            // reply_to directive: send reply first, then delete placeholder.
                            // Only delete if send succeeds — preserves placeholder on failure.
                            let mut send_ok = false;
                            let mut first = true;
                            for chunk in &chunks {
                                if first {
                                    match adapter.send_message_with_reply(
                                        &thread_channel,
                                        chunk,
                                        reply_id,
                                    ).await {
                                        Ok(_) => { send_ok = true; }
                                        Err(e) => {
                                            tracing::warn!(error = ?e, "reply_to send failed; preserving placeholder");
                                        }
                                    }
                                } else {
                                    let _ = adapter.send_message(&thread_channel, chunk).await;
                                }
                                first = false;
                            }
                            if send_ok {
                                if let Err(e) = adapter.delete_message(&msg).await {
                                    tracing::warn!(error = ?e, "delete placeholder failed; placeholder will remain visible");
                                }
                            }
                        } else if adapter.platform() == "discord"
                            && contains_bot_mention(&final_content)
                        {
                            // (#1112) The streamed reply mentions another bot but carries
                            // no [[reply_to]] directive. Editing it into the placeholder is
                            // a MESSAGE_UPDATE, which Discord does NOT emit a mention
                            // notification for (#1110) — so the mentioned peer would never
                            // wake. Delete the placeholder and send the chunk(s) as fresh
                            // messages so Discord emits MESSAGE_CREATE. (#1153 already
                            // propagated the mention to every chunk above.)
                            let mut send_ok = false;
                            if let Some(first) = chunks.first() {
                                if adapter.send_message(&thread_channel, first).await.is_ok() {
                                    send_ok = true;
                                }
                            }
                            for chunk in chunks.iter().skip(1) {
                                let _ = adapter.send_message(&thread_channel, chunk).await;
                            }
                            if send_ok {
                                if let Err(e) = adapter.delete_message(&msg).await {
                                    tracing::warn!(error = ?e, "delete placeholder failed; placeholder will remain visible");
                                }
                            }
                        } else {
                            // Normal streaming: edit first chunk into placeholder, send rest
                            if let Some(first) = chunks.first() {
                                let _ = adapter.edit_message(&msg, first).await;
                            }
                            for chunk in chunks.iter().skip(1) {
                                let _ = adapter.send_message(&thread_channel, chunk).await;
                            }
                        }
                    } else {
                        // Send-once: all chunks as new messages
                        // First chunk uses reply_to directive if present
                        let mut first = true;
                        for chunk in &chunks {
                            if first {
                                if let Some(ref reply_id) = directives.reply_to {
                                    let _ = adapter.send_message_with_reply(
                                        &thread_channel,
                                        chunk,
                                        reply_id,
                                    ).await;
                                } else {
                                    let _ = adapter.send_message(&thread_channel, chunk).await;
                                }
                            } else {
                                let _ = adapter.send_message(&thread_channel, chunk).await;
                            }
                            first = false;
                        }
                    }

                    Ok(())
                })
            })
            .await
    }
}

/// Extract all Discord mentions (`<@123>`, `<@!123>`, `<@&123>`) from content,
/// skipping mentions inside fenced code blocks (``` ... ```).
/// Normalizes `<@!UID>` to `<@UID>` for deduplication (same user).
/// Returns the deduplicated list in appearance order. (#1153)
fn extract_mentions(content: &str) -> Vec<String> {
    let mut mentions = Vec::new();
    let mut in_fence = false;

    for line in content.split('\n') {
        if line.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }

        let bytes = line.as_bytes();
        let mut i = 0;
        while i + 2 < bytes.len() {
            if bytes[i] == b'<' && bytes[i + 1] == b'@' {
                let (prefix_end, is_role) = if i + 2 < bytes.len() && bytes[i + 2] == b'&' {
                    (i + 3, true)
                } else if i + 2 < bytes.len() && bytes[i + 2] == b'!' {
                    (i + 3, false)
                } else {
                    (i + 2, false)
                };
                if prefix_end < bytes.len() && bytes[prefix_end].is_ascii_digit() {
                    if let Some(end) = line[prefix_end..].find('>') {
                        if line[prefix_end..prefix_end + end]
                            .chars()
                            .all(|c| c.is_ascii_digit())
                        {
                            // Normalize: <@!UID> → <@UID>, keep <@&RoleID> as-is
                            let uid = &line[prefix_end..prefix_end + end];
                            let normalized = if is_role {
                                format!("<@&{uid}>")
                            } else {
                                format!("<@{uid}>")
                            };
                            if !mentions.contains(&normalized) {
                                mentions.push(normalized);
                            }
                            i = prefix_end + end + 1;
                            continue;
                        }
                    }
                }
                i = prefix_end;
            } else {
                i += 1;
            }
        }
    }
    mentions
}

/// Compute the char length of the mention footer that will be appended
/// (`"\n" + mentions joined by " "`). Returns 0 if there are no mentions. (#1153)
fn mention_footer_len(mentions: &[String]) -> usize {
    if mentions.is_empty() {
        return 0;
    }
    1 + mentions.iter().map(|m| m.len()).sum::<usize>() + mentions.len().saturating_sub(1)
}

/// Append every mention to each split chunk that does not already contain it, so
/// receiving bots under a mention gate accept all pieces. `limit` is the hard
/// message-length ceiling; a chunk that would exceed it after appending is left
/// unchanged (the `mention_reserve` pre-deduction guarantees space in normal
/// cases). No-op for a single chunk or no mentions. (#1153)
fn propagate_mentions_to_chunks(
    chunks: Vec<String>,
    mentions: &[String],
    limit: usize,
) -> Vec<String> {
    if mentions.is_empty() || chunks.len() <= 1 {
        return chunks;
    }
    chunks
        .into_iter()
        .map(|chunk| {
            let missing: Vec<&String> = mentions
                .iter()
                .filter(|m| !chunk_contains_mention(&chunk, m))
                .collect();
            if missing.is_empty() {
                chunk
            } else {
                let footer = format!(
                    "\n{}",
                    missing
                        .iter()
                        .map(|m| m.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                if chunk.chars().count() + footer.chars().count() <= limit {
                    format!("{chunk}{footer}")
                } else {
                    // Safety: never exceed the hard limit.
                    chunk
                }
            }
        })
        .collect()
}

/// Check if a chunk contains an exact mention. Mentions are `<@DIGITS>`
/// terminated by `>`, so a substring search is exact — `<@123>` cannot match
/// inside `<@1234>` because `>` is the boundary delimiter. (#1153)
fn chunk_contains_mention(chunk: &str, mention: &str) -> bool {
    chunk.contains(mention)
}

/// Returns true if `content` contains a Discord user mention (`<@123>`,
/// `<@!123>`) or role mention (`<@&123>`). Used by the streaming path to switch
/// from edit (MESSAGE_UPDATE, no mention notification) to delete+send
/// (MESSAGE_CREATE) so a mentioned peer actually wakes. (#1112)
fn contains_bot_mention(content: &str) -> bool {
    let bytes = content.as_bytes();
    let mut i = 0;
    while i + 2 < bytes.len() {
        if bytes[i] == b'<' && bytes[i + 1] == b'@' {
            // Skip optional '!' (nickname mention) or '&' (role mention)
            let start =
                if i + 2 < bytes.len() && (bytes[i + 2] == b'!' || bytes[i + 2] == b'&') {
                    i + 3
                } else {
                    i + 2
                };
            if start < bytes.len() && bytes[start].is_ascii_digit() {
                if let Some(end) = content[start..].find('>') {
                    if content[start..start + end].chars().all(|c| c.is_ascii_digit()) {
                        return true;
                    }
                }
            }
            i = start;
        } else {
            i += 1;
        }
    }
    false
}

/// Flatten a tool-call title into a single line safe for inline-code spans.
fn sanitize_title(title: &str) -> String {
    title
        .replace('\r', "")
        .replace('\n', " ; ")
        .replace('`', "'")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolState {
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
struct ToolEntry {
    id: String,
    title: String,
    state: ToolState,
}

impl ToolEntry {
    fn render(&self) -> String {
        let icon = match self.state {
            ToolState::Running => "🔧",
            ToolState::Completed => "✅",
            ToolState::Failed => "❌",
        };
        let suffix = if self.state == ToolState::Running {
            "..."
        } else {
            ""
        };
        format!("{icon} `{}`{}", self.title, suffix)
    }
}

/// Maximum number of finished tool entries to show individually
/// during streaming before collapsing into a summary line.
const TOOL_COLLAPSE_THRESHOLD: usize = 3;

fn compose_display(
    tool_lines: &[ToolEntry],
    text: &str,
    streaming: bool,
    tool_display: ToolDisplay,
) -> String {
    let mut out = String::new();
    if !tool_lines.is_empty() && tool_display != ToolDisplay::None {
        let done = tool_lines
            .iter()
            .filter(|e| e.state == ToolState::Completed)
            .count();
        let failed = tool_lines
            .iter()
            .filter(|e| e.state == ToolState::Failed)
            .count();
        let running = tool_lines
            .iter()
            .filter(|e| e.state == ToolState::Running)
            .count();
        let finished = done + failed;

        match tool_display {
            ToolDisplay::Compact => {
                // Always show count summary, never per-tool details
                let mut parts = Vec::new();
                if done > 0 {
                    parts.push(format!("✅ {done}"));
                }
                if failed > 0 {
                    parts.push(format!("❌ {failed}"));
                }
                if running > 0 {
                    parts.push(format!("🔧 {running}"));
                }
                if !parts.is_empty() {
                    out.push_str(&format!("{} tool(s)\n", parts.join(" · ")));
                }
            }
            ToolDisplay::Full => {
                if streaming {
                    let running_entries: Vec<_> = tool_lines
                        .iter()
                        .filter(|e| e.state == ToolState::Running)
                        .collect();

                    if finished <= TOOL_COLLAPSE_THRESHOLD {
                        for entry in tool_lines.iter().filter(|e| e.state != ToolState::Running) {
                            out.push_str(&entry.render());
                            out.push('\n');
                        }
                    } else {
                        let mut parts = Vec::new();
                        if done > 0 {
                            parts.push(format!("✅ {done}"));
                        }
                        if failed > 0 {
                            parts.push(format!("❌ {failed}"));
                        }
                        out.push_str(&format!("{} tool(s) completed\n", parts.join(" · ")));
                    }

                    if running_entries.len() <= TOOL_COLLAPSE_THRESHOLD {
                        for entry in &running_entries {
                            out.push_str(&entry.render());
                            out.push('\n');
                        }
                    } else {
                        let hidden = running_entries.len() - TOOL_COLLAPSE_THRESHOLD;
                        out.push_str(&format!("🔧 {hidden} more running\n"));
                        for entry in running_entries.iter().skip(hidden) {
                            out.push_str(&entry.render());
                            out.push('\n');
                        }
                    }
                } else {
                    for entry in tool_lines {
                        out.push_str(&entry.render());
                        out.push('\n');
                    }
                }
            }
            ToolDisplay::None => {} // guarded above, but safe no-op
        }
        if !out.is_empty() {
            out.push('\n');
        }
    }
    out.push_str(text.trim_end());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time regression guard: use_streaming() is a required trait method
    /// (no default). Any adapter that forgets to implement it will fail to compile.
    /// This test documents the contract — see PR #503 / issue #502 for context.
    #[test]
    fn use_streaming_is_required_method() {
        // If use_streaming() had a default impl, this test module would still
        // compile even if an adapter forgot to override it. The real guard is
        // the trait definition itself — this test exists as documentation and
        // to catch if someone re-adds a default.
        struct TestAdapter;

        #[async_trait]
        impl ChatAdapter for TestAdapter {
            fn platform(&self) -> &'static str {
                "test"
            }
            fn message_limit(&self) -> usize {
                2000
            }
            async fn send_message(&self, _: &ChannelRef, _: &str) -> Result<MessageRef> {
                unimplemented!()
            }
            async fn create_thread(
                &self,
                _: &ChannelRef,
                _: &MessageRef,
                _: &str,
            ) -> Result<ChannelRef> {
                unimplemented!()
            }
            async fn add_reaction(&self, _: &MessageRef, _: &str) -> Result<()> {
                Ok(())
            }
            async fn remove_reaction(&self, _: &MessageRef, _: &str) -> Result<()> {
                Ok(())
            }
            // use_streaming() MUST be declared — removing this line should fail compilation
            fn use_streaming(&self, _other_bot_present: bool) -> bool {
                false
            }
        }

        let adapter = TestAdapter;
        // Verify the method is callable and returns the declared value
        assert!(!adapter.use_streaming(false));
    }

    #[test]
    fn origin_event_id_excluded_from_eq() {
        let a = ChannelRef {
            platform: "line".into(),
            channel_id: "U123".into(),
            thread_id: None,
            parent_id: None,
            origin_event_id: Some("evt_aaa".into()),
        };
        let b = ChannelRef {
            platform: "line".into(),
            channel_id: "U123".into(),
            thread_id: None,
            parent_id: None,
            origin_event_id: Some("evt_bbb".into()),
        };
        assert_eq!(a, b, "same channel with different event IDs must be equal");
    }

    #[test]
    fn origin_event_id_excluded_from_hash() {
        use std::collections::HashMap;
        let a = ChannelRef {
            platform: "line".into(),
            channel_id: "U123".into(),
            thread_id: None,
            parent_id: None,
            origin_event_id: Some("evt_aaa".into()),
        };
        let b = ChannelRef {
            platform: "line".into(),
            channel_id: "U123".into(),
            thread_id: None,
            parent_id: None,
            origin_event_id: Some("evt_bbb".into()),
        };
        let mut map = HashMap::new();
        map.insert(a, "first");
        // b should hit the same bucket and overwrite
        map.insert(b, "second");
        assert_eq!(map.len(), 1);
        assert_eq!(map.values().next(), Some(&"second"));
    }

    #[test]
    fn origin_event_id_survives_clone() {
        let ch = ChannelRef {
            platform: "line".into(),
            channel_id: "U123".into(),
            thread_id: None,
            parent_id: None,
            origin_event_id: Some("evt_abc".into()),
        };
        // Simulates create_thread propagation: clone preserves origin_event_id
        let thread_ch = ChannelRef {
            thread_id: Some("topic_1".into()),
            origin_event_id: ch.origin_event_id.clone(),
            ..ch.clone()
        };
        assert_eq!(thread_ch.origin_event_id.as_deref(), Some("evt_abc"));
    }

    fn tool(id: &str, title: &str, state: ToolState) -> ToolEntry {
        ToolEntry {
            id: id.into(),
            title: title.into(),
            state,
        }
    }

    #[test]
    fn compose_display_full_shows_complete_title() {
        let tools = vec![tool(
            "1",
            "curl -s https://example.com",
            ToolState::Completed,
        )];
        let out = compose_display(&tools, "done", false, ToolDisplay::Full);
        assert!(out.contains("`curl -s https://example.com`"));
    }

    #[test]
    fn compose_display_compact_shows_count_summary() {
        let tools = vec![
            tool("1", "curl -s https://example.com", ToolState::Completed),
            tool("2", "grep -r pattern src/", ToolState::Completed),
            tool("3", "cat /etc/hosts", ToolState::Failed),
        ];
        let out = compose_display(&tools, "done", false, ToolDisplay::Compact);
        assert!(out.contains("✅ 2"), "expected completed count: {out}");
        assert!(out.contains("❌ 1"), "expected failed count: {out}");
        assert!(out.contains("tool(s)"), "expected tool(s) label: {out}");
        // Must NOT contain individual tool names
        assert!(!out.contains("curl"), "should not show tool names: {out}");
        assert!(!out.contains("grep"), "should not show tool names: {out}");
    }

    #[test]
    fn compose_display_compact_shows_running_count() {
        let tools = vec![
            tool("1", "curl", ToolState::Completed),
            tool("2", "npm install", ToolState::Running),
        ];
        let out = compose_display(&tools, "", true, ToolDisplay::Compact);
        assert!(out.contains("✅ 1"), "expected completed count: {out}");
        assert!(out.contains("🔧 1"), "expected running count: {out}");
    }

    #[test]
    fn compose_display_none_hides_tools() {
        let tools = vec![tool(
            "1",
            "curl -s https://example.com",
            ToolState::Completed,
        )];
        let out = compose_display(&tools, "response text", false, ToolDisplay::None);
        assert_eq!(out, "response text");
    }
}

#[cfg(test)]
mod directive_tests {
    use crate::config::{AgentAttachmentsConfig, WorkspaceRequest};
    use std::fs;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use super::{
        contains_bot_mention, extract_mentions, mention_footer_len, parse_output_directives,
        prepare_output_attachments, propagate_mentions_to_chunks, AdapterRouter, ChannelRef,
        ChatAdapter, MessageRef, OutboundAttachment,
    };

    fn attachment_config(root: &Path) -> AgentAttachmentsConfig {
        AgentAttachmentsConfig {
            enabled: true,
            allowed_paths: vec![root.to_string_lossy().to_string()],
            max_files: 5,
            max_file_bytes: 64,
            max_total_bytes: 128,
        }
    }

    #[test]
    fn parse_reply_to_directive() {
        let input = "[[reply_to:1502606076451885136]]\nHello world";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("1502606076451885136".to_string()));
        assert!(directives.attachments.is_empty());
        assert_eq!(content, "Hello world");
    }

    #[test]
    fn parse_no_directives() {
        let input = "Just plain content\nwith multiple lines";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, None);
        assert_eq!(content, input);
    }

    #[test]
    fn parse_multiple_directives() {
        let input = "[[reply_to:123456]]\n[[unknown_key:value]]\nContent here";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("123456".to_string()));
        assert_eq!(content, "Content here");
    }

    #[test]
    fn parse_invalid_reply_to_rejects_whitespace() {
        let input = "[[reply_to:has spaces]]\nContent";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, None);
        assert_eq!(content, "Content");
    }

    #[test]
    fn parse_slack_ts_format_accepted() {
        let input = "[[reply_to:1234567890.123456]]\nContent";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("1234567890.123456".to_string()));
        assert_eq!(content, "Content");
    }

    #[test]
    fn parse_empty_reply_to() {
        let input = "[[reply_to:]]\nContent";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, None);
        assert_eq!(content, "Content");
    }

    #[test]
    fn parse_crlf_line_endings() {
        let input = "[[reply_to:999]]\r\nContent with CRLF";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("999".to_string()));
        assert_eq!(content, "Content with CRLF");
    }

    #[test]
    fn parse_directive_only_no_content() {
        let input = "[[reply_to:123]]";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("123".to_string()));
        assert_eq!(content, "");
    }

    #[test]
    fn parse_late_reply_to_after_plain_text() {
        let input = "Normal first line\n[[reply_to:123]]\nMore content";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("123".to_string()));
        assert_eq!(content, "Normal first line\nMore content");
    }

    #[test]
    fn parse_late_reply_to_inside_handoff_text() {
        let input = "我会使用 handoff 技能。Automatic approval review approved.[[reply_to:1512261692329824328]] <@1511955018805280849> 请继承当前 workspace。";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("1512261692329824328".to_string()));
        assert_eq!(
            content,
            "我会使用 handoff 技能。Automatic approval review approved. <@1511955018805280849> 请继承当前 workspace。"
        );
    }

    #[test]
    fn parse_invalid_late_reply_to_is_preserved() {
        let input = "Explain [[reply_to:has spaces]] syntax";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, None);
        assert_eq!(content, input);
    }

    #[test]
    fn parse_duplicate_reply_to_last_wins() {
        let input = "[[reply_to:111]]\n[[reply_to:222]]\nContent";
        let (directives, content) = parse_output_directives(input);
        // Last value wins
        assert_eq!(directives.reply_to, Some("222".to_string()));
        assert_eq!(content, "Content");
    }

    #[test]
    fn parse_crlf_multiple_directives() {
        let input = "[[reply_to:456]]\r\n[[unknown:x]]\r\nContent after CRLF";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("456".to_string()));
        assert_eq!(content, "Content after CRLF");
    }

    #[test]
    fn parse_bracket_without_colon_preserved() {
        // [[Note]] has no colon — not a directive, preserved as content
        let input = "[[Summary]]\nThis is body text";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, None);
        assert_eq!(content, input);
    }

    #[test]
    fn parse_reply_to_with_inline_content() {
        // Agent puts content on same line as directive — should still parse
        let input = "[[reply_to:1502724086474870926]]  @BOT I'm on standby";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("1502724086474870926".to_string()));
        assert_eq!(content, "@BOT I'm on standby");
    }

    #[test]
    fn parse_reply_to_inline_with_more_lines() {
        let input = "[[reply_to:123]]  First line\nSecond line\nThird line";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("123".to_string()));
        assert_eq!(content, "First line\nSecond line\nThird line");
    }

    #[test]
    fn parse_reply_to_no_space_before_content() {
        // No space between ]] and content
        let input = "[[reply_to:1502724086474870926]]收到";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("1502724086474870926".to_string()));
        assert_eq!(content, "收到");
    }

    #[test]
    fn parse_reply_to_inline_with_mention() {
        // Real-world case: directive followed by Discord mention
        let input = "[[reply_to:1502724086474870926]]  <@1490365068863606784> 我 standby";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("1502724086474870926".to_string()));
        assert_eq!(content, "<@1490365068863606784> 我 standby");
    }

    #[test]
    fn parse_reply_to_inline_only_spaces() {
        // Trailing spaces only — no real content, should be empty
        let input = "[[reply_to:123]]   ";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("123".to_string()));
        assert_eq!(content, "");
    }

    #[test]
    fn parse_reply_to_with_brackets_in_content() {
        // Content after ]] contains brackets — should not confuse parser
        let input = "[[reply_to:456]]  看看 [[這個]] 怎麼樣";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("456".to_string()));
        assert_eq!(content, "看看 [[這個]] 怎麼樣");
    }

    #[test]
    fn parse_attach_directive() {
        let input = "[[attach:/workspace/out.log]]\nHere it is";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.attachments, vec!["/workspace/out.log"]);
        assert_eq!(content, "Here it is");
    }

    #[test]
    fn parse_multiple_attach_directives() {
        let input = "[[attach:a.log]]\n[[attach:b.png]]\nDone";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.attachments, vec!["a.log", "b.png"]);
        assert_eq!(content, "Done");
    }

    #[test]
    fn parse_reply_to_and_attach_inline() {
        let input = "[[reply_to:123]] [[attach:shot.png]] Screenshot attached";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.reply_to, Some("123".to_string()));
        assert_eq!(directives.attachments, vec!["shot.png"]);
        assert_eq!(content, "Screenshot attached");
    }

    #[test]
    fn late_attach_is_preserved_as_text() {
        let input = "Explain [[attach:foo.log]] syntax";
        let (directives, content) = parse_output_directives(input);
        assert!(directives.attachments.is_empty());
        assert_eq!(content, input);
    }

    #[test]
    fn parse_late_attach_after_agent_preamble() {
        let input = "Tool finished.[[attach:/workspace/out.log]]\nDone.";
        let (directives, content) = parse_output_directives(input);
        assert_eq!(directives.attachments, vec!["/workspace/out.log"]);
        assert_eq!(content, "Tool finished.\nDone.");
    }

    #[test]
    fn parse_late_attach_only_at_line_boundary_after_close() {
        let input = "Explain [[attach:/workspace/out.log]] syntax";
        let (directives, content) = parse_output_directives(input);
        assert!(directives.attachments.is_empty());
        assert_eq!(content, input);
    }

    #[test]
    fn prepare_attachment_resolves_relative_from_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("project");
        fs::create_dir(&project).unwrap();
        fs::write(project.join("hello.log"), b"hello").unwrap();
        let config = attachment_config(tmp.path());

        let (attachments, warnings) =
            prepare_output_attachments(&["hello.log".into()], &config, &project);

        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].filename, "hello.log");
        assert_eq!(attachments[0].bytes, b"hello");
    }

    #[test]
    fn prepare_attachment_accepts_absolute_allowlisted_path() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("result.txt");
        fs::write(&file, b"ok").unwrap();
        let config = attachment_config(tmp.path());

        let (attachments, warnings) =
            prepare_output_attachments(&[file.to_string_lossy().to_string()], &config, tmp.path());

        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].filename, "result.txt");
    }

    #[test]
    fn prepare_attachment_rejects_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config = attachment_config(tmp.path());

        let (attachments, warnings) =
            prepare_output_attachments(&["missing.log".into()], &config, tmp.path());

        assert!(attachments.is_empty());
        assert!(warnings.iter().any(|w| w.contains("missing.log")));
    }

    #[test]
    fn prepare_attachment_rejects_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("dir");
        fs::create_dir(&dir).unwrap();
        let config = attachment_config(tmp.path());

        let (attachments, warnings) =
            prepare_output_attachments(&["dir".into()], &config, tmp.path());

        assert!(attachments.is_empty());
        assert!(warnings.iter().any(|w| w.contains("not a regular file")));
    }

    #[cfg(unix)]
    #[test]
    fn prepare_attachment_rejects_unreadable_file() {
        if unsafe { libc::geteuid() } == 0 {
            return;
        }
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("locked.log");
        fs::write(&file, b"locked").unwrap();
        fs::set_permissions(&file, fs::Permissions::from_mode(0o000)).unwrap();
        let config = attachment_config(tmp.path());

        let (attachments, warnings) =
            prepare_output_attachments(&["locked.log".into()], &config, tmp.path());

        fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(attachments.is_empty());
        assert!(warnings.iter().any(|w| w.contains("cannot be read")));
    }

    #[test]
    fn prepare_attachment_rejects_outside_allowlist() {
        let tmp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("outside.log");
        fs::write(&outside_file, b"no").unwrap();
        let config = attachment_config(tmp.path());

        let (attachments, warnings) = prepare_output_attachments(
            &[outside_file.to_string_lossy().to_string()],
            &config,
            tmp.path(),
        );

        assert!(attachments.is_empty());
        assert!(warnings
            .iter()
            .any(|w| w.contains("outside the configured allowlist")));
    }

    #[cfg(unix)]
    #[test]
    fn prepare_attachment_rejects_symlink_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("secret.log");
        fs::write(&outside_file, b"no").unwrap();
        std::os::unix::fs::symlink(&outside_file, tmp.path().join("link.log")).unwrap();
        let config = attachment_config(tmp.path());

        let (attachments, warnings) =
            prepare_output_attachments(&["link.log".into()], &config, tmp.path());

        assert!(attachments.is_empty());
        assert!(warnings
            .iter()
            .any(|w| w.contains("outside the configured allowlist")));
    }

    #[test]
    fn prepare_attachment_rejects_too_many_files() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = attachment_config(tmp.path());
        config.max_files = 1;

        let (attachments, warnings) =
            prepare_output_attachments(&["a.log".into(), "b.log".into()], &config, tmp.path());

        assert!(attachments.is_empty());
        assert!(warnings.iter().any(|w| w.contains("too many files")));
    }

    #[test]
    fn prepare_attachment_rejects_per_file_oversize() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("big.log"), vec![b'x'; 65]).unwrap();
        let config = attachment_config(tmp.path());

        let (attachments, warnings) =
            prepare_output_attachments(&["big.log".into()], &config, tmp.path());

        assert!(attachments.is_empty());
        assert!(warnings.iter().any(|w| w.contains("too large")));
    }

    #[test]
    fn prepare_attachment_rejects_total_oversize() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("a.log"), vec![b'a'; 64]).unwrap();
        fs::write(tmp.path().join("b.log"), vec![b'b'; 64]).unwrap();
        fs::write(tmp.path().join("c.log"), vec![b'c'; 1]).unwrap();
        let config = attachment_config(tmp.path());

        let (attachments, warnings) = prepare_output_attachments(
            &["a.log".into(), "b.log".into(), "c.log".into()],
            &config,
            tmp.path(),
        );

        assert!(attachments.is_empty());
        assert!(warnings.iter().any(|w| w.contains("total attachment size")));
    }

    #[test]
    fn prepare_attachment_rejects_when_disabled() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("a.log"), b"a").unwrap();
        let mut config = attachment_config(tmp.path());
        config.enabled = false;

        let (attachments, warnings) =
            prepare_output_attachments(&["a.log".into()], &config, tmp.path());

        assert!(attachments.is_empty());
        assert!(warnings.iter().any(|w| w.contains("disabled")));
    }

    #[derive(Default)]
    struct FallbackAdapter {
        sent: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl ChatAdapter for FallbackAdapter {
        fn platform(&self) -> &'static str {
            "test"
        }

        fn message_limit(&self) -> usize {
            2000
        }

        async fn send_message(
            &self,
            channel: &ChannelRef,
            content: &str,
        ) -> anyhow::Result<MessageRef> {
            self.sent.lock().unwrap().push(content.to_string());
            Ok(MessageRef {
                channel: channel.clone(),
                message_id: "1".into(),
            })
        }

        async fn create_thread(
            &self,
            channel: &ChannelRef,
            _trigger_msg: &MessageRef,
            _title: &str,
        ) -> anyhow::Result<ChannelRef> {
            Ok(channel.clone())
        }

        async fn add_reaction(&self, _msg: &MessageRef, _emoji: &str) -> anyhow::Result<()> {
            Ok(())
        }

        async fn remove_reaction(&self, _msg: &MessageRef, _emoji: &str) -> anyhow::Result<()> {
            Ok(())
        }

        fn use_streaming(&self, _other_bot_present: bool) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn non_discord_attachment_fallback_warns() {
        let adapter = FallbackAdapter::default();
        let channel = ChannelRef {
            platform: "test".into(),
            channel_id: "c".into(),
            thread_id: None,
            parent_id: None,
            origin_event_id: None,
        };
        let attachments = vec![OutboundAttachment {
            filename: "a.log".into(),
            bytes: b"a".to_vec(),
        }];

        adapter
            .send_message_with_attachments(&channel, "body", &attachments, None)
            .await
            .unwrap();

        let sent = adapter.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert!(sent[0].contains("body"));
        assert!(sent[0].contains("attachment upload is not supported"));
    }

    #[test]
    fn parse_session_directives_inline_ws_title() {
        let (directives, content) =
            AdapterRouter::parse_session_directives("[[ws:foo]] [[title:修 bug]]  do work")
                .unwrap();
        assert_eq!(
            directives.workspace,
            Some(WorkspaceRequest::Existing("foo".to_string()))
        );
        assert_eq!(directives.title, Some("修 bug".to_string()));
        assert_eq!(content, "do work");
    }

    #[test]
    fn parse_session_directives_multiline_and_create() {
        let (directives, content) =
            AdapterRouter::parse_session_directives("[[ws:team/foo --create]]\n開始調查").unwrap();
        assert_eq!(
            directives.workspace,
            Some(WorkspaceRequest::Create("team/foo".to_string()))
        );
        assert_eq!(content, "開始調查");
    }

    #[test]
    fn parse_session_directives_duplicate_key_last_wins() {
        let (directives, content) = AdapterRouter::parse_session_directives(
            "[[ws:foo]] [[ws:bar]] [[title:old]] [[title:new]] work",
        )
        .unwrap();
        assert_eq!(
            directives.workspace,
            Some(WorkspaceRequest::Existing("bar".to_string()))
        );
        assert_eq!(directives.title, Some("new".to_string()));
        assert_eq!(content, "work");
    }

    #[test]
    fn parse_session_directives_absent_preserves_prompt() {
        let input = "do work in /workspace/foo";
        let (directives, content) = AdapterRouter::parse_session_directives(input).unwrap();
        assert_eq!(directives.workspace, None);
        assert_eq!(directives.title, None);
        assert_eq!(content, input);
    }

    #[test]
    fn parse_session_directives_old_cwd_syntax_is_plain_prompt() {
        let input = "[cwd:/workspace/foo]\ndo work";
        let (directives, content) = AdapterRouter::parse_session_directives(input).unwrap();
        assert_eq!(directives.workspace, None);
        assert_eq!(directives.title, None);
        assert_eq!(content, input);
    }

    #[test]
    fn parse_session_directives_unknown_key_errors() {
        let err = AdapterRouter::parse_session_directives("[[wz:foo]] do work").unwrap_err();
        assert!(err.to_string().contains("unknown session directive: wz"));
    }

    #[test]
    fn parse_session_directives_unknown_flag_errors() {
        let err = AdapterRouter::parse_session_directives("[[ws:foo --init]] do work").unwrap_err();
        assert!(err.to_string().contains("unknown workspace flag: --init"));
    }

    #[test]
    fn parse_session_directives_empty_values_error() {
        let err = AdapterRouter::parse_session_directives("[[ws:]] do work").unwrap_err();
        assert!(err.to_string().contains("empty workspace directive"));

        let err = AdapterRouter::parse_session_directives("[[title:]] do work").unwrap_err();
        assert!(err.to_string().contains("empty title directive"));
    }

    #[test]
    fn parse_session_directives_rejects_unterminated() {
        let err = AdapterRouter::parse_session_directives("[[ws:foo").unwrap_err();
        assert!(err.to_string().contains("unterminated session directive"));
    }

    #[test]
    fn parse_session_directives_rejects_malformed() {
        let err = AdapterRouter::parse_session_directives("[[ws]] do work").unwrap_err();
        assert!(err.to_string().contains("malformed session directive"));
    }

    // --- (#1153) mention extraction + propagation across split chunks ---

    #[test]
    fn extract_mentions_basic() {
        assert_eq!(
            extract_mentions("hello <@123> and <@&456> world"),
            vec!["<@123>", "<@&456>"]
        );
    }

    #[test]
    fn extract_mentions_dedup() {
        assert_eq!(extract_mentions("<@123> foo <@123> bar"), vec!["<@123>"]);
    }

    #[test]
    fn extract_mentions_normalizes_nickname() {
        assert_eq!(extract_mentions("hey <@!789>"), vec!["<@789>"]);
    }

    #[test]
    fn extract_mentions_dedup_after_normalize() {
        // <@123> and <@!123> are the same user
        assert_eq!(extract_mentions("<@123> and <@!123>"), vec!["<@123>"]);
    }

    #[test]
    fn extract_mentions_skips_code_blocks() {
        let content = "hello <@111>\n```\n<@222>\n```\nworld <@333>";
        assert_eq!(extract_mentions(content), vec!["<@111>", "<@333>"]);
    }

    #[test]
    fn extract_mentions_role_vs_user_distinct() {
        assert_eq!(
            extract_mentions("<@&999> and <@999>"),
            vec!["<@&999>", "<@999>"]
        );
    }

    #[test]
    fn extract_mentions_none() {
        assert!(extract_mentions("no mentions; email user@example.com").is_empty());
    }

    #[test]
    fn mention_footer_len_values() {
        assert_eq!(mention_footer_len(&[]), 0);
        // "\n<@123>" = 1 + 6
        assert_eq!(mention_footer_len(&["<@123>".to_string()]), 7);
        // "\n<@123> <@456>" = 1 + 6 + 1 + 6
        assert_eq!(
            mention_footer_len(&["<@123>".to_string(), "<@456>".to_string()]),
            14
        );
    }

    #[test]
    fn propagate_mentions_single_chunk_noop() {
        let chunks = vec!["hello <@123>".to_string()];
        assert_eq!(
            propagate_mentions_to_chunks(chunks.clone(), &["<@123>".to_string()], 2000),
            chunks
        );
    }

    #[test]
    fn propagate_mentions_appends_to_missing_chunks() {
        let chunks = vec!["part1 <@123>".to_string(), "part2 no mention".to_string()];
        let out = propagate_mentions_to_chunks(chunks, &["<@123>".to_string()], 2000);
        assert_eq!(out[0], "part1 <@123>");
        assert_eq!(out[1], "part2 no mention\n<@123>");
    }

    #[test]
    fn propagate_mentions_respects_hard_limit() {
        // chunk already at limit: appending would exceed it, so leave unchanged.
        let chunk = "x".repeat(2000);
        let chunks = vec!["<@1> head".to_string(), chunk.clone()];
        let out = propagate_mentions_to_chunks(chunks, &["<@1>".to_string()], 2000);
        assert_eq!(out[1], chunk); // unchanged — no footer busting the 2000 ceiling
    }

    #[test]
    fn pipeline_split_then_propagate() {
        // End-to-end: a long mention-bearing message splits, and every chunk
        // carries the mention after propagation.
        let mention = "<@1514992231969456258>";
        let body = format!("{mention} {}", "word ".repeat(800)); // ~4000+ chars
        let mentions = extract_mentions(&body);
        let reserve = mention_footer_len(&mentions);
        let chunks = crate::format::split_message(&body, 2000usize.saturating_sub(reserve));
        let chunks = propagate_mentions_to_chunks(chunks, &mentions, 2000);
        assert!(chunks.len() >= 2, "expected the body to split");
        for c in &chunks {
            assert!(c.contains(mention), "every chunk must carry the mention");
            assert!(c.chars().count() <= 2000, "no chunk exceeds the hard limit");
        }
    }

    // --- (#1112) bot-mention detection for the streaming wake path ---

    #[test]
    fn contains_bot_mention_user() {
        assert!(contains_bot_mention("hello <@1234567890> world"));
    }

    #[test]
    fn contains_bot_mention_nickname() {
        assert!(contains_bot_mention("hey <@!9876543210>"));
    }

    #[test]
    fn contains_bot_mention_role() {
        assert!(contains_bot_mention("calling <@&1496247626675257384>"));
    }

    #[test]
    fn contains_bot_mention_embedded() {
        assert!(contains_bot_mention("请问 <@1501788608439386172> 1+1=?"));
    }

    #[test]
    fn contains_bot_mention_no_match() {
        assert!(!contains_bot_mention("hello world"));
        assert!(!contains_bot_mention("email user@example.com"));
        assert!(!contains_bot_mention("<@not_a_number>"));
    }
}
