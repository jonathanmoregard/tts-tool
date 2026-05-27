# tts-tool

CleanText on stdin -> MP3 on stdout (or `-o FILE`) via Fish Audio TTS.

```
echo "Hello world." | tts-tool -o hello.mp3
substack-url-tool "$URL" | tts-tool -o article.mp3
```

## Voice cloning

Three flows, all wrapping Fish Audio's `voices.create`:

```sh
# A. From an existing recording:
tts-tool clone --sample me.wav --title "My voice"

# B. Inline capture (single clip, pw-record / arecord / ffmpeg):
tts-tool clone --record 60 --title "My voice"

# C. Interactive multi-segment (recommended for expressive clones):
tts-tool record-clone --title "My voice" --speaker-name "Jonathan"

# Anonymous (no name in the script):
tts-tool record-clone --title "Some-voice"
```

`record-clone` walks you through six register-varied segments (warmup,
tender, playful, reflective, stability, intimate). Stop-on-ENTER per
segment, then (p)lay / (y)es / (r)edo / (s)kip / (q)uit. Bundles
all accepted takes into one upload with transcripts — gives the
embedding phonetic + prosodic breadth without forcing a single
4-minute take. Script is person-neutral; `--speaker-name` fills in
the warmup introduction, or omit it for anonymous cloning.

All three print the new Fish Audio `reference_id`. Plug into synth via
`--voice-id ID` or `export FISH_AUDIO_VOICE_ID=ID`.

## Install (dev)

```sh
nix develop          # devShell with uv + python + ffmpeg
uv sync --all-extras
uv run pytest
uv run tts-tool --help
```

## Install (Nix flake)

```sh
nix run github:jonathanmoregard/tts-tool -- -o out.mp3 < text.txt
nix profile install github:jonathanmoregard/tts-tool
```

On dellan, the tool is wired into `environment.systemPackages` and reads its
API key from `config.age.secrets.fish-audio-api-key.path` via a wrapper.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `FISH_AUDIO_API_KEY` | (required) | Fish Audio API key |
| `FISH_AUDIO_API_KEY_FILE` | — | Path to file containing the key (overrides above) |
| `FISH_AUDIO_MODEL` | `s2-pro` | Backend: `s2-pro`, `s1` |
| `FISH_AUDIO_VOICE_ID` | (SDK default) | Voice library `reference_id` |
| `LISTEN_SPEED` | `1.0` | Prosody speed, 0.8 - 1.2 |
| `LISTEN_CACHE_DIR` | platformdirs cache | Chunk MP3 cache location |

## How it works

```
stdin (CleanText: title, blank, body w/ \n\n paragraphs)
  -> spaCy sentence segmentation (en_core_web_sm parser)
  -> greedy pack into <=2000-char chunks (never split mid-sentence)
  -> per chunk:
       cache key = sha256(text | model | voice_id | speed)
       hit: load cached mp3 ; miss: Fish Audio synth + write cache
  -> ffmpeg concat -c copy -> single MP3
```

The content-hash cache means re-runs of the same article cost nothing,
and a partial run that fails midway resumes free.

## Clear cache

```sh
rm -rf "$(python -c 'from platformdirs import user_cache_dir; print(user_cache_dir("tts-tool"))')"
```

## Known limits (v0.1)

- English only.
- Cloud TTS (Fish Audio) only. No self-hosted backend.
- No multi-speaker, no streaming playback.
- No RSS / podcast feed generation (separate tool, later).
- `ffmpeg` must be on PATH (Nix flake handles this).
