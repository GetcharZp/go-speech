package pipertts

import (
	"encoding/json"
	"os"
	"strings"
)

// loadPiperConfig 读取 .onnx.json 配置文件
func loadPiperConfig(path string) (PiperConfig, error) {
	var cfg PiperConfig
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// getInitial 提取声母逻辑
func getInitial(py string) string {
	// 优先匹配双字母声母
	for _, s := range []string{"zh", "ch", "sh"} {
		if strings.HasPrefix(py, s) {
			return s
		}
	}
	// 匹配单字母声母
	initials := "b p m f d t n l g k h j q x r z c s y w"
	if len(py) > 0 {
		firstChar := string(py[0])
		if strings.Contains(initials, firstChar) {
			return firstChar
		}
	}
	return ""
}
