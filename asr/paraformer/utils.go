package paraformer

import (
	"bufio"
	"fmt"
	"github.com/up-zero/gotool/mediautil"
	"os"
	"strconv"
	"strings"
)

// loadTokens 加载 Token ID 映射表
//
// 数据格式: token id
func loadTokens(path string) (map[int]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	m := make(map[int]string)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			token := parts[0]
			idStr := parts[1]
			id, err := strconv.Atoi(idStr)
			if err == nil {
				m[id] = token
			}
		}
	}
	return m, scanner.Err()
}

// loadCMVN 解析 am.mvn 文件
// 返回 neg_mean (均值的负数) 和 inv_std (标准差的倒数)
func loadCMVN(path string) ([]float32, []float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	var negMean, invStd []float32
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "<LearnRateCoef>") {
			continue
		}

		// 读取数据并转换为 float32
		parts := strings.Fields(line)
		values := make([]float32, 0, 80)
		dataParts := parts[3 : len(parts)-1]
		for _, v := range dataParts {
			fVal, err := strconv.ParseFloat(v, 32)
			if err != nil {
				continue
			}
			values = append(values, float32(fVal))
		}

		if negMean == nil {
			negMean = values
		} else {
			invStd = values
			break
		}
	}

	if len(negMean) == 0 || len(invStd) == 0 {
		return nil, nil, fmt.Errorf("未找到有效的 CMVN 数据")
	}
	return negMean, invStd, nil
}

// parseWavBytes 转换 WAV 字节流
func parseWavBytes(wavBytes []byte) ([]float32, error) {
	// 音频格式转换
	targetBytes, err := mediautil.ReformatWavBytes(wavBytes, sampleRate, channels, bitsPerSample)
	if err != nil {
		return nil, fmt.Errorf("无法格式化 WAV 文件: %v", err)
	}
	return mediautil.PcmBytesToFloat32(targetBytes[44:], bitsPerSample)
}
