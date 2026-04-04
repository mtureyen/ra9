<script lang="ts">
  import { lastDebug, brainHistory } from "./stores";

  let copyLabel = $state("Copy Brain");

  function formatDebug(d: any, index?: number): string {
    const header = index !== undefined ? `─── Exchange ${index + 1} ───` : "";
    const lines = [
      header,
      `  amygdala: ${d.final_emotion} (${(d.final_intensity || 0).toFixed(2)})${d.reappraised ? ` → reappraised from ${d.fast_emotion}` : ""}`,
      `  recalled: ${d.recalled_count || 0} memories${d.recalled_top ? ` (top: "${d.recalled_top}")` : ""}`,
      d.mood ? `  mood: ${Object.entries(d.mood).filter(([_, v]) => Math.abs((v as number) - 0.5) > 0.03).map(([k, v]) => `${k.replace("_", " ")}=${(v as number).toFixed(2)}`).join(", ")}` : null,
      `  inner voice: ${d.inner_voice_nudge || "—"}`,
      d.inner_speech ? `  inner speech: "${d.inner_speech}"` : d.system2_bypassed ? "  inner speech: bypassed (warmth)" : null,
      d.social_perception ? `  social perception: ${d.social_perception}` : null,
      `  prediction error: ${(d.prediction_error || 0).toFixed(2)}`,
      `  embodied: energy=${((d.embodied_energy || 0) * 100).toFixed(0)}% comfort=${((d.embodied_comfort || 0) * 100).toFixed(0)}%`,
      d.tone_alignment !== undefined ? `  tone alignment: ${d.tone_alignment.toFixed(2)}` : null,
      d.encoded ? "  encoded: yes" : null,
      d.discovery ? "  discovery: detected" : null,
      d.dmn_flash ? "  dmn: spontaneous thought" : null,
      d.anamnesis ? `  retrieval: ${d.anamnesis.strategy}${d.anamnesis.detected_person ? ` → ${d.anamnesis.detected_person}` : ''}` : null,
      d.anamnesis ? `  familiarity: ${(d.anamnesis.familiarity || 0).toFixed(2)}  recollection: ${(d.anamnesis.recollection || 0).toFixed(2)}` : null,
      d.anamnesis ? `  effort: ${(d.anamnesis.effort || 0).toFixed(2)}  iterations: ${d.anamnesis.iterations_used || 0}  candidates: ${d.anamnesis.total_candidates || 0} → ${d.anamnesis.conscious_count || 0}` : null,
      d.anamnesis?.tot_active ? `  TOT: active (${d.anamnesis.tot_partial_person || '?'}, ${d.anamnesis.tot_partial_emotion || '?'})` : null,
      d.anamnesis?.narrative ? `  narrative: "${d.anamnesis.narrative}"` : null,
      d.anamnesis?.priming_words?.length ? `  priming: ${d.anamnesis.priming_words.join(', ')}` : null,
    ].filter(Boolean).join("\n");
    return lines;
  }

  async function copyBrain() {
    if ($brainHistory.length === 0 && !$lastDebug) return;

    // Copy full session brain history
    const allEntries = $brainHistory.length > 0
      ? $brainHistory.map((d, i) => formatDebug(d, i)).join("\n\n")
      : formatDebug($lastDebug!);

    await navigator.clipboard.writeText(allEntries);
    copyLabel = "Copied!";
    setTimeout(() => copyLabel = "Copy Brain", 1500);
  }

  function emotionColor(emotion: string): string {
    const colors: Record<string, string> = {
      joy: "var(--joy)",
      sadness: "var(--sadness)",
      anger: "var(--anger)",
      fear: "var(--fear)",
      trust: "var(--trust)",
      awe: "var(--awe)",
      surprise: "var(--surprise)",
      disgust: "var(--disgust)",
    };
    return colors[emotion] || "var(--text-secondary)";
  }
</script>

<div class="monitor">
  <div class="header">
    <h3>Brain Monitor</h3>
    {#if $lastDebug}
      <button class="copy-btn" onclick={copyBrain}>{copyLabel}</button>
    {/if}
  </div>

  {#if $lastDebug}
    <div class="section">
      <div class="row">
        <span class="label">Emotion</span>
        <span class="value" style="color: {emotionColor($lastDebug.final_emotion || '')}">
          {$lastDebug.final_emotion || "—"} ({($lastDebug.final_intensity || 0).toFixed(2)})
        </span>
      </div>
      {#if $lastDebug.reappraised}
        <div class="note">reappraised from {$lastDebug.fast_emotion}</div>
      {/if}
    </div>

    <div class="section">
      <div class="row">
        <span class="label">Inner Voice</span>
        <span class="pill">{$lastDebug.inner_voice_nudge || "—"}</span>
      </div>
    </div>

    {#if $lastDebug.inner_speech}
      <div class="section highlight">
        <div class="label">Inner Speech (System 2)</div>
        <div class="speech">"{$lastDebug.inner_speech}"</div>
      </div>
    {:else if $lastDebug.system2_bypassed}
      <div class="section">
        <div class="row">
          <span class="label">Inner Speech</span>
          <span class="muted">bypassed (warmth)</span>
        </div>
      </div>
    {/if}

    {#if $lastDebug.social_perception}
      <div class="section">
        <div class="row">
          <span class="label">Reading You</span>
          <span class="value">{$lastDebug.social_perception}</span>
        </div>
      </div>
    {/if}

    <div class="section">
      <div class="row">
        <span class="label">Prediction Error</span>
        <span class="value">{($lastDebug.prediction_error || 0).toFixed(2)}</span>
      </div>
    </div>

    <div class="section">
      <div class="row">
        <span class="label">Energy</span>
        <div class="bar-container">
          <div
            class="bar energy"
            style="width: {($lastDebug.embodied_energy || 0) * 100}%"
          ></div>
        </div>
        <span class="value-small">{(($lastDebug.embodied_energy || 0) * 100).toFixed(0)}%</span>
      </div>
    </div>

    <div class="section">
      <div class="row">
        <span class="label">Comfort</span>
        <div class="bar-container">
          <div
            class="bar comfort"
            style="width: {($lastDebug.embodied_comfort || 0) * 100}%"
          ></div>
        </div>
        <span class="value-small">{(($lastDebug.embodied_comfort || 0) * 100).toFixed(0)}%</span>
      </div>
    </div>

    {#if $lastDebug.tone_alignment !== undefined}
      <div class="section">
        <div class="row">
          <span class="label">Tone Alignment</span>
          <span class="value">{($lastDebug.tone_alignment).toFixed(2)}</span>
        </div>
      </div>
    {/if}

    <div class="section">
      <div class="row">
        <span class="label">Recalled</span>
        <span class="value">{$lastDebug.recalled_count || 0} memories</span>
      </div>
      {#if $lastDebug.recalled_top}
        <div class="note truncate">top: {$lastDebug.recalled_top}</div>
      {/if}
    </div>

    {#if $lastDebug.anamnesis}
      {@const a = $lastDebug.anamnesis}
      <div class="section highlight">
        <div class="row">
          <span class="label">Strategy</span>
          <span class="pill strategy">{a.strategy}{a.detected_person ? ` → ${a.detected_person}` : ''}</span>
        </div>
      </div>

      <div class="section">
        <div class="row">
          <span class="label">Familiarity</span>
          <div class="bar-container">
            <div class="bar familiarity" style="width: {(a.familiarity || 0) * 100}%"></div>
          </div>
          <span class="value-small">{((a.familiarity || 0) * 100).toFixed(0)}%</span>
        </div>
        <div class="row">
          <span class="label">Recollection</span>
          <div class="bar-container">
            <div class="bar recollection" style="width: {(a.recollection || 0) * 100}%"></div>
          </div>
          <span class="value-small">{((a.recollection || 0) * 100).toFixed(0)}%</span>
        </div>
      </div>

      <div class="section">
        <div class="row">
          <span class="label">Effort</span>
          <div class="bar-container">
            <div class="bar effort" style="width: {(a.effort || 0) * 100}%"></div>
          </div>
          <span class="value-small">{((a.effort || 0) * 100).toFixed(0)}%</span>
        </div>
        <div class="note" style="padding-left: 98px">
          {a.iterations_used || 0} iterations · {a.total_candidates || 0} candidates → {a.conscious_count || 0} conscious
        </div>
      </div>

      {#if a.tot_active}
        <div class="section highlight">
          <div class="row">
            <span class="label">Tip of Tongue</span>
            <span class="indicator warning">active</span>
          </div>
          {#if a.tot_partial_person || a.tot_partial_emotion}
            <div class="note" style="padding-left: 98px">
              partial: {a.tot_partial_person || '?'}, {a.tot_partial_emotion || '?'}
            </div>
          {/if}
        </div>
      {/if}

      {#if a.source_confusions && a.source_confusions.length > 0}
        <div class="section">
          <div class="row">
            <span class="label">Source Confusion</span>
            <span class="indicator warning">{a.source_confusions.length} detected</span>
          </div>
        </div>
      {/if}

      {#if a.narrative}
        <div class="section highlight">
          <div class="label">Narrative</div>
          <div class="speech">{a.narrative}</div>
        </div>
      {/if}

      {#if a.priming_words && a.priming_words.length > 0}
        <div class="section">
          <div class="label" style="margin-bottom: 4px">Priming</div>
          <div class="priming-words">
            {#each a.priming_words as word}
              <span class="priming-pill">{word}</span>
            {/each}
          </div>
        </div>
      {/if}

      {#if a.prospective_triggers && a.prospective_triggers.length > 0}
        {#each a.prospective_triggers as trigger}
          <div class="section">
            <div class="row">
              <span class="label">Intention</span>
              <span class="indicator accent">{trigger}</span>
            </div>
          </div>
        {/each}
      {/if}
    {/if}

    <div class="section indicators">
      {#if $lastDebug.encoded}
        <span class="indicator success">encoded</span>
      {/if}
      {#if $lastDebug.discovery}
        <span class="indicator accent">discovery</span>
      {/if}
      {#if $lastDebug.dmn_flash}
        <span class="indicator warning">dmn flash</span>
      {/if}
    </div>
  {:else}
    <div class="empty">
      <p class="muted">Waiting for first exchange...</p>
    </div>
  {/if}
</div>

<style>
  .monitor {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  h3 {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
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

  .section {
    padding: 6px 8px;
    border-radius: 4px;
  }

  .section.highlight {
    background: var(--bg-panel);
    border: 1px solid var(--border);
  }

  .row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .label {
    font-size: 11px;
    color: var(--text-muted);
    min-width: 90px;
    flex-shrink: 0;
  }

  .value {
    font-size: 13px;
    color: var(--text-primary);
  }

  .value-small {
    font-size: 11px;
    color: var(--text-secondary);
    min-width: 32px;
    text-align: right;
  }

  .muted {
    color: var(--text-muted);
    font-size: 12px;
  }

  .pill {
    padding: 2px 10px;
    border-radius: 12px;
    background: var(--accent);
    color: white;
    font-size: 11px;
    font-weight: 500;
  }

  .speech {
    font-size: 12px;
    color: var(--text-secondary);
    font-style: italic;
    margin-top: 4px;
    line-height: 1.4;
  }

  .note {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
    padding-left: 98px;
  }

  .truncate {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 300px;
  }

  .bar-container {
    flex: 1;
    height: 6px;
    background: var(--bg-input);
    border-radius: 3px;
    overflow: hidden;
  }

  .bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .bar.energy {
    background: linear-gradient(to right, var(--danger), var(--warning), var(--success));
  }

  .bar.comfort {
    background: var(--trust);
  }

  .indicators {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }

  .indicator {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .indicator.success {
    background: rgba(34, 197, 94, 0.15);
    color: var(--success);
  }

  .indicator.accent {
    background: rgba(99, 102, 241, 0.15);
    color: var(--accent);
  }

  .indicator.warning {
    background: rgba(245, 158, 11, 0.15);
    color: var(--warning);
  }

  .empty {
    padding: 40px 12px;
    text-align: center;
  }

  .pill.strategy {
    background: var(--trust);
  }

  .bar.familiarity {
    background: var(--accent);
  }

  .bar.recollection {
    background: var(--joy);
  }

  .bar.effort {
    background: linear-gradient(to right, var(--success), var(--warning), var(--danger));
  }

  .priming-words {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }

  .priming-pill {
    padding: 1px 6px;
    border-radius: 8px;
    background: rgba(99, 102, 241, 0.1);
    color: var(--text-muted);
    font-size: 10px;
  }
</style>
