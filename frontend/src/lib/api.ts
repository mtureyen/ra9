const API_BASE = "http://127.0.0.1:8000";

export interface AnamnesisDebug {
  strategy: string;
  detected_person: string | null;
  is_recall_query: boolean;
  familiarity: number;
  recollection: number;
  tot_active: boolean;
  tot_partial_person: string | null;
  tot_partial_emotion: string | null;
  effort: number;
  total_candidates: number;
  conscious_count: number;
  unconscious_pool_size: number;
  source_confusions: Array<Record<string, unknown>>;
  iterations_used: number;
  priming_words: string[];
  narrative: string | null;
  prospective_triggers: string[];
}

export interface DebugDict {
  fast_emotion?: string;
  fast_intensity?: number;
  reappraised?: boolean;
  final_emotion?: string;
  final_intensity?: number;
  inner_voice_nudge?: string;
  inner_speech?: string;
  social_perception?: string;
  prediction_error?: number;
  embodied_energy?: number;
  embodied_comfort?: number;
  tone_alignment?: number;
  discovery?: boolean;
  dmn_flash?: boolean;
  system2_bypassed?: boolean;
  recalled_count?: number;
  recalled_top?: string;
  encoded?: boolean;
  mood?: Record<string, number>;
  anamnesis?: AnamnesisDebug;
  [key: string]: unknown;
}

export interface SessionBootResponse {
  session_id: string;
  self_schema: string;
}

export interface MoodState {
  novelty_seeking: number;
  social_bonding: number;
  analytical_depth: number;
  playfulness: number;
  caution: number;
  expressiveness: number;
}

export interface EmbodiedState {
  energy: number;
  cognitive_load: number;
  comfort: number;
}

export async function bootSession(): Promise<SessionBootResponse> {
  const res = await fetch(`${API_BASE}/session/boot`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function endSession(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/session/end`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function* streamChat(
  message: string
): AsyncGenerator<{ type: "chunk"; text: string } | { type: "done"; debug: DebugDict } | { type: "error"; message: string }> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!res.ok) throw new Error(await res.text());
  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          yield data;
        } catch {
          // Skip malformed SSE lines
        }
      }
    }
  }
}

export async function getMood(): Promise<MoodState> {
  const res = await fetch(`${API_BASE}/mood`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getEmbodied(): Promise<EmbodiedState> {
  const res = await fetch(`${API_BASE}/embodied`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getHealth(): Promise<{ status: string; brain: boolean; session: boolean }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
