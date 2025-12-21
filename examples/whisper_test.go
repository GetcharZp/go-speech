package examples

import (
	"fmt"
	"github.com/getcharzp/go-speech/asr/whisper"
	"testing"
)

func TestWhisper(t *testing.T) {
	cfg := whisper.DefaultConfig()
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"
	cfg.EncoderModelPath = "../whisper_weights/small_encoder_model.onnx"
	cfg.DecoderModelPath = "../whisper_weights/small_decoder_model_merged.onnx"
	cfg.TokensPath = "../whisper_weights/vocab.json"
	cfg.AddedTokensPath = "../whisper_weights/added_tokens.json"

	asrEngine, err := whisper.NewEngine(cfg)
	if err != nil {
		t.Fatalf("创建引擎失败: %v", err)
	}
	defer asrEngine.Destroy()

	text, err := asrEngine.TranscribeFile("./zh-en.wav", whisper.TranscribeOption{
		Language: whisper.LangZh,
		Task:     whisper.TaskTranscribe,
	})
	if err != nil {
		t.Fatalf("识别出错: %v", err)
		return
	}
	fmt.Printf("识别结果: %s\n", text)
}
