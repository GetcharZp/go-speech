package examples

import (
	"fmt"
	"github.com/getcharzp/go-speech/asr/paraformer"
	"testing"
)

func TestParaformer(t *testing.T) {
	config := paraformer.Config{
		OnnxRuntimeLibPath:    "../lib/onnxruntime.dll",
		ModelPath:             "../paraformer_weights/model.int8.onnx",
		TokensPath:            "../paraformer_weights/tokens.txt",
		CMVNPath:              "../paraformer_weights/am.mvn",
		PunctuationModelPath:  "../paraformer_weights/punctuation_model.onnx",
		PunctuationTokensPath: "../paraformer_weights/punctuation_tokens.json",
	}

	asrEngine, err := paraformer.NewEngine(config)
	if err != nil {
		t.Fatalf("创建引擎失败: %v", err)
	}
	defer asrEngine.Destroy()

	text, err := asrEngine.TranscribeFile("./zh-en.wav")
	if err != nil {
		t.Fatalf("识别出错: %v", err)
		return
	}
	fmt.Printf("识别结果: %s\n", text)
}
