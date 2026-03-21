# @inferrlm/react-native-mlx

Run LLMs, Text-to-Speech, and Speech-to-Text on-device in React Native using [MLX Swift](https://github.com/ml-explore/mlx-swift).

Built with [Nitro Modules](https://github.com/mrousavy/nitro) for zero-overhead native bridging.

## Requirements

- iOS 16.0+
- React Native 0.76+
- react-native-nitro-modules

## Installation

```bash
npm install @inferrlm/react-native-mlx react-native-nitro-modules
cd ios && pod install
```

## Usage

### Download a Model

```typescript
import { ModelManager, MLXModel } from '@inferrlm/react-native-mlx'

await ModelManager.download(MLXModel.Qwen3_1_7B_4bit, (progress) => {
  console.log(`${(progress * 100).toFixed(1)}%`)
})
```

### Load and Generate

```typescript
import { LLM } from '@inferrlm/react-native-mlx'

await LLM.load('mlx-community/Qwen3-1.7B-4bit', {
  onProgress: (p) => console.log(`Loading: ${(p * 100).toFixed(0)}%`),
  manageHistory: true,
})

const response = await LLM.generate('What is the capital of France?')
```

### Streaming

```typescript
let response = ''
await LLM.stream('Tell me a story', (token) => {
  response += token
})
```

### Streaming with Events

For thinking blocks (chain-of-thought) and tool calls, use the event-based API:

```typescript
await LLM.streamWithEvents('Solve 2+3 step by step', (event) => {
  switch (event.type) {
    case 'thinking_start':
      showThinkingIndicator()
      break
    case 'thinking_chunk':
      appendToThinking(event.chunk)
      break
    case 'thinking_end':
      hideThinkingIndicator()
      break
    case 'token':
      appendToContent(event.token)
      break
    case 'generation_end':
      console.log(`${event.stats.tokensPerSecond} tok/s`)
      break
  }
})
```

### Configuring Generation

```typescript
LLM.systemPrompt = 'You are a helpful coding assistant.'
LLM.maxTokens = 2048
LLM.temperature = 0.7
LLM.enableThinking = true
```

### Tool Calling

```typescript
import { LLM, createTool } from '@inferrlm/react-native-mlx'
import { z } from 'zod'

const weatherTool = createTool({
  name: 'get_weather',
  description: 'Get weather for a city',
  arguments: z.object({
    city: z.string().describe('City name'),
  }),
  handler: async ({ city }) => {
    return { temperature: 22, condition: 'sunny' }
  },
})

await LLM.load('mlx-community/Qwen3-1.7B-4bit', {
  onProgress: (p) => console.log(`${(p * 100).toFixed(0)}%`),
  tools: [weatherTool],
})

await LLM.stream('What is the weather in Tokyo?', (token) => {
  process.stdout.write(token)
}, (update) => {
  console.log(`Tool called: ${update.toolCall.name}`)
})
```

### Conversation History

When `manageHistory: true` is set during load, conversation turns are automatically tracked:

```typescript
await LLM.load('mlx-community/Qwen3-1.7B-4bit', {
  manageHistory: true,
})

await LLM.stream('My name is Alice', onToken)
await LLM.stream('What is my name?', onToken) // Remembers "Alice"

const history = LLM.getHistory()
LLM.clearHistory()
```

### Stop Generation

```typescript
LLM.stop()
```

### Unload

```typescript
LLM.unload()
```

### Text-to-Speech

```typescript
import { TTS, MLXModel } from '@inferrlm/react-native-mlx'

await TTS.load(MLXModel.PocketTTS, {
  onProgress: (p) => console.log(`${(p * 100).toFixed(0)}%`),
})

const audioBuffer = await TTS.generate('Hello world!', {
  voice: 'alba',
  speed: 1.0,
})

await TTS.stream('Hello world!', (chunk) => {
  // Process each audio chunk
}, { voice: 'alba' })
```

Available voices: `alba`, `azelma`, `cosette`, `eponine`, `fantine`, `javert`, `jean`, `marius`

### Speech-to-Text

```typescript
import { STT, MLXModel } from '@inferrlm/react-native-mlx'

await STT.load(MLXModel.GLM_ASR_Nano_4bit, {
  onProgress: (p) => console.log(`${(p * 100).toFixed(0)}%`),
})

const text = await STT.transcribe(audioBuffer)

// Live microphone transcription
await STT.startListening()
const partial = await STT.transcribeBuffer()
const final = await STT.stopListening()
```

## API

### LLM

#### Methods

| Method | Description |
|--------|-------------|
| `load(modelId, options?)` | Load a model into memory. Downloads from HuggingFace if not cached |
| `generate(prompt)` | Generate a complete response (blocking) |
| `stream(prompt, onToken, onToolCall?)` | Stream tokens with optional tool call updates |
| `streamWithEvents(prompt, onEvent)` | Stream with typed events for thinking/tool calls |
| `stop()` | Stop current generation |
| `unload()` | Unload model and free memory |
| `getLastGenerationStats()` | Get token count, speed, timing from last generation |
| `getHistory()` | Get conversation history (if `manageHistory` enabled) |
| `clearHistory()` | Clear conversation history |

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `isLoaded` | `boolean` | - | Whether a model is loaded (read-only) |
| `isGenerating` | `boolean` | - | Whether generation is in progress (read-only) |
| `modelId` | `string` | `''` | Currently loaded model ID (read-only) |
| `debug` | `boolean` | `false` | Enable debug logging |
| `systemPrompt` | `string` | `'You are a helpful assistant.'` | System prompt for the model |
| `maxTokens` | `number` | `2048` | Maximum tokens to generate |
| `temperature` | `number` | `0.7` | Sampling temperature (0 = deterministic) |
| `enableThinking` | `boolean` | `true` | Enable thinking mode for supported models |

#### LLMLoadOptions

| Property | Type | Description |
|----------|------|-------------|
| `onProgress` | `(progress: number) => void` | Loading progress callback (0-1) |
| `additionalContext` | `LLMMessage[]` | Conversation history or few-shot examples |
| `manageHistory` | `boolean` | Automatically manage conversation history |
| `tools` | `ToolDefinition[]` | Tools available for the model to call |

#### GenerationStats

| Property | Type | Description |
|----------|------|-------------|
| `tokenCount` | `number` | Total tokens generated |
| `tokensPerSecond` | `number` | Generation speed |
| `timeToFirstToken` | `number` | Time to first token (ms) |
| `totalTime` | `number` | Total generation time (ms) |
| `toolExecutionTime` | `number` | Time spent executing tools (ms) |

#### StreamEvent Types

| Event | Fields | Description |
|-------|--------|-------------|
| `generation_start` | `timestamp` | Generation began |
| `token` | `token` | Response token |
| `thinking_start` | `timestamp` | Model began thinking |
| `thinking_chunk` | `chunk` | Thinking content chunk |
| `thinking_end` | `content`, `timestamp` | Thinking complete |
| `tool_call_start` | `id`, `name`, `arguments` | Tool call initiated |
| `tool_call_executing` | `id` | Tool handler running |
| `tool_call_completed` | `id`, `result` | Tool returned result |
| `tool_call_failed` | `id`, `error` | Tool execution failed |
| `generation_end` | `content`, `stats` | Generation complete |

### TTS

#### Methods

| Method | Description |
|--------|-------------|
| `load(modelId, options?)` | Load a TTS model |
| `generate(text, options?)` | Generate audio buffer from text |
| `stream(text, onAudioChunk, options?)` | Stream audio chunks |
| `stop()` | Stop generation |
| `unload()` | Unload model |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `isLoaded` | `boolean` | Whether a TTS model is loaded |
| `isGenerating` | `boolean` | Whether audio is being generated |
| `modelId` | `string` | Currently loaded model ID |
| `sampleRate` | `number` | Audio sample rate (e.g. 24000) |

#### TTSGenerateOptions

| Property | Type | Description |
|----------|------|-------------|
| `voice` | `string` | Voice name |
| `speed` | `number` | Speech speed multiplier |

### STT

#### Methods

| Method | Description |
|--------|-------------|
| `load(modelId, options?)` | Load an STT model |
| `transcribe(audio)` | Transcribe an audio buffer |
| `transcribeStream(audio, onToken)` | Stream transcription tokens |
| `startListening()` | Start microphone capture |
| `transcribeBuffer()` | Transcribe current buffer while listening |
| `stopListening()` | Stop listening and get final transcript |
| `stop()` | Stop transcription |
| `unload()` | Unload model |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `isLoaded` | `boolean` | Whether an STT model is loaded |
| `isTranscribing` | `boolean` | Whether transcription is in progress |
| `isListening` | `boolean` | Whether the microphone is active |
| `modelId` | `string` | Currently loaded model ID |

### ModelManager

#### Methods

| Method | Description |
|--------|-------------|
| `download(modelId, onProgress)` | Download a model from HuggingFace |
| `isDownloaded(modelId)` | Check if a model is downloaded |
| `getDownloadedModels()` | Get list of downloaded model IDs |
| `deleteModel(modelId)` | Delete a downloaded model |
| `getModelPath(modelId)` | Get local filesystem path |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `debug` | `boolean` | Enable debug logging |

## Supported Models

Any MLX-compatible model from [mlx-community](https://huggingface.co/mlx-community) should work. The `MLXModel` enum provides pre-tested models:

```typescript
import { MLXModel } from '@inferrlm/react-native-mlx'
```

### LLM Models

| Model | Enum Key | HuggingFace ID |
|-------|----------|----------------|
| **Llama 3.2 (Meta)** | | |
| Llama 3.2 1B 4-bit | `Llama_3_2_1B_Instruct_4bit` | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| Llama 3.2 1B 8-bit | `Llama_3_2_1B_Instruct_8bit` | `mlx-community/Llama-3.2-1B-Instruct-8bit` |
| Llama 3.2 3B 4-bit | `Llama_3_2_3B_Instruct_4bit` | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Llama 3.2 3B 8-bit | `Llama_3_2_3B_Instruct_8bit` | `mlx-community/Llama-3.2-3B-Instruct-8bit` |
| **Qwen 2.5 (Alibaba)** | | |
| Qwen 2.5 0.5B 4-bit | `Qwen2_5_0_5B_Instruct_4bit` | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Qwen 2.5 0.5B 8-bit | `Qwen2_5_0_5B_Instruct_8bit` | `mlx-community/Qwen2.5-0.5B-Instruct-8bit` |
| Qwen 2.5 1.5B 4-bit | `Qwen2_5_1_5B_Instruct_4bit` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Qwen 2.5 1.5B 8-bit | `Qwen2_5_1_5B_Instruct_8bit` | `mlx-community/Qwen2.5-1.5B-Instruct-8bit` |
| Qwen 2.5 3B 4-bit | `Qwen2_5_3B_Instruct_4bit` | `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| Qwen 2.5 3B 8-bit | `Qwen2_5_3B_Instruct_8bit` | `mlx-community/Qwen2.5-3B-Instruct-8bit` |
| **Qwen 3** | | |
| Qwen 3 1.7B 4-bit | `Qwen3_1_7B_4bit` | `mlx-community/Qwen3-1.7B-4bit` |
| Qwen 3 1.7B 8-bit | `Qwen3_1_7B_8bit` | `mlx-community/Qwen3-1.7B-8bit` |
| **Qwen 3.5** | | |
| Qwen 3.5 0.8B 4-bit | `Qwen3_5_0_8B_MLX_4bit` | `mlx-community/Qwen3.5-0.8B-MLX-4bit` |
| Qwen 3.5 0.8B 8-bit | `Qwen3_5_0_8B_MLX_8bit` | `mlx-community/Qwen3.5-0.8B-MLX-8bit` |
| **Gemma 3 (Google)** | | |
| Gemma 3 1B 4-bit | `Gemma_3_1B_IT_4bit` | `mlx-community/gemma-3-1b-it-4bit` |
| Gemma 3 1B 8-bit | `Gemma_3_1B_IT_8bit` | `mlx-community/gemma-3-1b-it-8bit` |
| **Phi 3.5 Mini (Microsoft)** | | |
| Phi 3.5 Mini 4-bit | `Phi_3_5_Mini_Instruct_4bit` | `mlx-community/Phi-3.5-mini-instruct-4bit` |
| Phi 3.5 Mini 8-bit | `Phi_3_5_Mini_Instruct_8bit` | `mlx-community/Phi-3.5-mini-instruct-8bit` |
| **Phi 4 Mini (Microsoft)** | | |
| Phi 4 Mini 4-bit | `Phi_4_Mini_Instruct_4bit` | `mlx-community/Phi-4-mini-instruct-4bit` |
| Phi 4 Mini 8-bit | `Phi_4_Mini_Instruct_8bit` | `mlx-community/Phi-4-mini-instruct-8bit` |
| **SmolLM (HuggingFace)** | | |
| SmolLM 1.7B 4-bit | `SmolLM_1_7B_Instruct_4bit` | `mlx-community/SmolLM-1.7B-Instruct-4bit` |
| SmolLM 1.7B 8-bit | `SmolLM_1_7B_Instruct_8bit` | `mlx-community/SmolLM-1.7B-Instruct-8bit` |
| **SmolLM2 (HuggingFace)** | | |
| SmolLM2 1.7B 4-bit | `SmolLM2_1_7B_Instruct_4bit` | `mlx-community/SmolLM2-1.7B-Instruct-4bit` |
| SmolLM2 1.7B 8-bit | `SmolLM2_1_7B_Instruct_8bit` | `mlx-community/SmolLM2-1.7B-Instruct-8bit` |
| **OpenELM (Apple)** | | |
| OpenELM 1.1B 4-bit | `OpenELM_1_1B_4bit` | `mlx-community/OpenELM-1_1B-4bit` |
| OpenELM 1.1B 8-bit | `OpenELM_1_1B_8bit` | `mlx-community/OpenELM-1_1B-8bit` |
| OpenELM 3B 4-bit | `OpenELM_3B_4bit` | `mlx-community/OpenELM-3B-4bit` |
| OpenELM 3B 8-bit | `OpenELM_3B_8bit` | `mlx-community/OpenELM-3B-8bit` |

### TTS Models

| Model | Enum Key | HuggingFace ID |
|-------|----------|----------------|
| **PocketTTS (Kyutai)** - 44.6M params | | |
| PocketTTS bf16 | `PocketTTS` | `mlx-community/pocket-tts` |
| PocketTTS 8-bit | `PocketTTS_8bit` | `mlx-community/pocket-tts-8bit` |
| PocketTTS 4-bit | `PocketTTS_4bit` | `mlx-community/pocket-tts-4bit` |

### STT Models

| Model | Enum Key | HuggingFace ID |
|-------|----------|----------------|
| **GLM-ASR (Alibaba)** | | |
| GLM-ASR Nano 4-bit | `GLM_ASR_Nano_4bit` | `mlx-community/GLM-ASR-Nano-2512-4bit` |

Browse more models at [huggingface.co/mlx-community](https://huggingface.co/mlx-community).

## License

MIT
