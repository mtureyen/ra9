<script lang="ts">
  import { streamChat } from "./api";
  import { messages, isStreaming, lastDebug, brainHistory, mood, embodied } from "./stores";
  import type { Message } from "./stores";

  let inputText = $state("");
  let messagesEl: HTMLDivElement;
  let copyLabel = $state("Copy Chat");

  async function copyChat() {
    const text = $messages.map(m => {
      const who = m.role === "user" ? "You" : "Ryo";
      return `${who}: ${m.content}`;
    }).join("\n\n");
    await navigator.clipboard.writeText(text);
    copyLabel = "Copied!";
    setTimeout(() => copyLabel = "Copy Chat", 1500);
  }

  function scrollToBottom() {
    if (messagesEl) {
      setTimeout(() => {
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }, 10);
    }
  }

  async function send() {
    const text = inputText.trim();
    if (!text || $isStreaming) return;

    inputText = "";

    // Add user message
    const userMsg: Message = { role: "user", content: text, timestamp: Date.now() };
    $messages = [...$messages, userMsg];
    scrollToBottom();

    // Start streaming
    $isStreaming = true;
    let assistantContent = "";
    const assistantMsg: Message = {
      role: "assistant",
      content: "",
      timestamp: Date.now(),
    };
    $messages = [...$messages, assistantMsg];

    try {
      for await (const event of streamChat(text)) {
        if (event.type === "chunk") {
          assistantContent += event.text;
          // Update last message in place
          $messages = [
            ...$messages.slice(0, -1),
            { ...assistantMsg, content: assistantContent },
          ];
          scrollToBottom();
        } else if (event.type === "done") {
          $lastDebug = event.debug;
          $brainHistory = [...$brainHistory, event.debug];
          if (event.debug.mood) {
            $mood = event.debug.mood as any;
          }
          if (event.debug.embodied_energy !== undefined) {
            $embodied = {
              energy: event.debug.embodied_energy as number,
              cognitive_load: 0,
              comfort: (event.debug.embodied_comfort as number) || 0.5,
            };
          }
          // Attach debug to the assistant message
          $messages = [
            ...$messages.slice(0, -1),
            { ...assistantMsg, content: assistantContent, debug: event.debug },
          ];
        } else if (event.type === "error") {
          assistantContent = `[Error: ${event.message}]`;
          $messages = [
            ...$messages.slice(0, -1),
            { ...assistantMsg, content: assistantContent },
          ];
        }
      }
    } catch (e: any) {
      assistantContent = `[Connection error: ${e.message}]`;
      $messages = [
        ...$messages.slice(0, -1),
        { ...assistantMsg, content: assistantContent },
      ];
    } finally {
      $isStreaming = false;
      scrollToBottom();
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      send();
    }
  }
</script>

<div class="chat">
  <div class="messages" bind:this={messagesEl}>
    {#each $messages as msg}
      <div class="message {msg.role}">
        <span class="label">{msg.role === "user" ? "You" : "Ryo"}</span>
        <div class="content">{msg.content}</div>
      </div>
    {/each}
    {#if $isStreaming && $messages[$messages.length - 1]?.content === ""}
      <div class="thinking">thinking...</div>
    {/if}
  </div>

  <div class="toolbar">
    {#if $messages.length > 0}
      <button class="copy-btn" onclick={copyChat}>{copyLabel}</button>
    {/if}
  </div>

  <div class="input-area">
    <textarea
      bind:value={inputText}
      onkeydown={handleKeydown}
      placeholder="Type a message... (Ctrl+Enter to send)"
      rows={3}
      disabled={$isStreaming}
    ></textarea>
    <button onclick={send} disabled={$isStreaming || !inputText.trim()}>
      Send
    </button>
  </div>
</div>

<style>
  .chat {
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .message {
    max-width: 85%;
  }

  .message.user {
    align-self: flex-end;
  }

  .message.assistant {
    align-self: flex-start;
  }

  .label {
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 2px;
    display: block;
  }

  .message.user .label {
    text-align: right;
  }

  .content {
    padding: 8px 12px;
    border-radius: var(--radius);
    white-space: pre-wrap;
    word-wrap: break-word;
    line-height: 1.6;
  }

  .message.user .content {
    background: var(--accent);
    color: white;
  }

  .message.assistant .content {
    background: var(--bg-panel);
    border: 1px solid var(--border);
  }

  .thinking {
    color: var(--text-muted);
    font-style: italic;
    padding: 8px 12px;
  }

  .toolbar {
    display: flex;
    justify-content: flex-end;
    padding: 4px 16px 0;
    min-height: 24px;
  }

  .copy-btn {
    padding: 2px 10px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-muted);
    font-family: var(--font);
    font-size: 10px;
    cursor: pointer;
  }

  .copy-btn:hover {
    border-color: var(--accent);
    color: var(--text-secondary);
  }

  .input-area {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 8px;
    background: var(--bg-secondary);
  }

  textarea {
    flex: 1;
    padding: 8px 12px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 13px;
    resize: none;
    outline: none;
  }

  textarea:focus {
    border-color: var(--accent);
  }

  textarea:disabled {
    opacity: 0.5;
  }

  button {
    padding: 8px 20px;
    background: var(--accent);
    border: none;
    border-radius: var(--radius);
    color: white;
    font-family: var(--font);
    font-size: 13px;
    cursor: pointer;
    align-self: flex-end;
  }

  button:hover:not(:disabled) {
    background: var(--accent-dim);
  }

  button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
</style>
