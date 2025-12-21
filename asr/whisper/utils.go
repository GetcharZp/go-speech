package whisper

import (
	"encoding/json"
	"fmt"
	"github.com/up-zero/gotool/mediautil"
	"os"
	"sync"
)

var (
	byteDecoder map[rune]byte
	decoderOnce sync.Once
)

func calculateNumHeads(layers int) int {
	switch layers {
	case 4:
		return 6
	case 6:
		return 8
	case 12:
		return 12
	case 24:
		return 16
	case 32:
		return 20
	default:
		return 8
	}
}

// loadTokens 加载 Token ID 映射表
func loadTokens(vPath, aPath string) (map[int]string, map[string]int, error) {
	tm := make(map[int]string)

	// 加载 vocab.json 文件
	vf, err := os.ReadFile(vPath)
	if err != nil {
		return nil, nil, err
	}
	var v map[string]int
	if err := json.Unmarshal(vf, &v); err != nil {
		return nil, nil, err
	}
	for s, id := range v {
		tm[id] = s
	}

	// 加载 added_tokens.json 文件
	af, err := os.ReadFile(aPath)
	if err != nil {
		return nil, nil, err
	}
	var a map[string]int
	if err := json.Unmarshal(af, &a); err != nil {
		return nil, nil, err
	}
	for s, id := range a {
		tm[id] = s
	}

	return tm, a, nil
}

// initByteDecoder 初始化字节 Decoder
func initByteDecoder() {
	decoderOnce.Do(func() {
		byteDecoder = make(map[rune]byte)
		for i := '!'; i <= '~'; i++ {
			byteDecoder[i] = byte(i)
		}
		for i := rune(161); i <= rune(172); i++ {
			byteDecoder[i] = byte(i)
		}
		for i := rune(174); i <= rune(255); i++ {
			byteDecoder[i] = byte(i)
		}
		n := 0
		for b := 0; b < 256; b++ {
			if (b >= int('!') && b <= int('~')) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255) {
				continue
			}
			byteDecoder[rune(256+n)] = byte(b)
			n++
		}
	})
}

// parseWavBytes 转换 WAV 字节流
func parseWavBytes(wavBytes []byte) ([]float32, error) {
	targetBytes, err := mediautil.ReformatWavBytes(wavBytes, sampleRate, channels, bitsPerSample)
	if err != nil {
		return nil, fmt.Errorf("无法格式化 WAV 文件: %v", err)
	}
	return mediautil.PcmBytesToFloat32(targetBytes[44:], 16)
}
