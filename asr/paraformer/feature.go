package paraformer

import (
	"fmt"
	"github.com/up-zero/gotool/mediautil"
	"math"
	"sync"
)

var (
	window     []float32
	melFilters [][]float32
	once       sync.Once
)

// extractFeatures 特征处理
//
// 流程: Wave -> FilterBank -> LFR -> CMVN
func (p *Engine) extractFeatures(samples []float32) ([]float32, int32, error) {
	const (
		melBins = 80
		lfrM    = 7 // Window size
		lfrN    = 6 // Window shift
	)

	// 提取 FilterBank
	fBankData, numFrames := computeFilterBank(samples, sampleRate, melBins)
	if numFrames == 0 {
		return nil, 0, fmt.Errorf("FBank特征提取失败: 帧数小于 1")
	}

	// 应用 LFR (Low Frame Rate)
	lfrData, lfrFrames := applyLFR(fBankData, numFrames, melBins, lfrM, lfrN)
	if lfrFrames == 0 {
		return nil, 0, fmt.Errorf("LFR特征提取失败: 帧数小于 1")
	}

	// CMVN
	if len(p.negMean) > 0 && len(p.invStd) > 0 {
		mediautil.ApplyCMVN(lfrData, p.negMean, p.invStd)
	}

	// 展平为一维数组
	totalLen := lfrFrames * (melBins * lfrM)
	flattened := make([]float32, totalLen)
	rowSize := melBins * lfrM
	for i, frame := range lfrData {
		copy(flattened[i*rowSize:], frame)
	}

	return flattened, int32(lfrFrames), nil
}

// computeFilterBank 计算 FilterBank 特征
func computeFilterBank(samples []float32, sampleRate int, melBins int) ([][]float32, int) {
	const (
		frameLen   = 400 // 25ms @ 16kHz
		frameShift = 160 // 10ms @ 16kHz
		fftSize    = 512 // Next power of 2
	)
	once.Do(func() {
		window = mediautil.HammingWindow(frameLen)
		melFilters = mediautil.MelFilters(sampleRate, fftSize, melBins, 0, 0)
	})

	// 预加重
	emphasized := mediautil.PreEmphasis(samples, 0.97)

	// 准备基础数据
	numSamples := len(emphasized)
	if numSamples < frameLen {
		return nil, 0
	}

	// 计算帧数
	numFrames := (numSamples-frameLen)/frameShift + 1

	// 分配结果矩阵
	features := make([][]float32, numFrames)
	// 预分配复数 buffer
	fftBuffer := make([]complex128, fftSize)

	for i := 0; i < numFrames; i++ {
		start := i * frameShift

		// 加窗 & 填充 FFT buffer
		for j := 0; j < fftSize; j++ {
			if j < frameLen {
				val := emphasized[start+j] * window[j]
				fftBuffer[j] = complex(float64(val), 0)
			} else {
				fftBuffer[j] = 0 // 补零
			}
		}

		// FFT 变换
		spectrum := mediautil.FFT(fftBuffer)

		// 计算 Mel 能量
		features[i] = make([]float32, melBins)
		for k := 0; k < melBins; k++ {
			sum := 0.0
			// 遍历 FFT 结果的前半部分 (Nyquist)
			for j := 0; j < fftSize/2+1; j++ {
				w := melFilters[k][j]
				if w > 0 {
					// Power = |X|^2
					r := real(spectrum[j])
					im := imag(spectrum[j])
					power := r*r + im*im
					sum += power * float64(w)
				}
			}

			if sum < 1e-7 {
				sum = 1e-7
			}
			features[i][k] = float32(math.Log(sum))
		}
	}

	return features, numFrames
}

// applyLFR (Low Frame Rate)
func applyLFR(inputs [][]float32, numFrames int, inputDim int, lfrM int, lfrN int) ([][]float32, int) {
	if numFrames < lfrM {
		return nil, 0
	}

	// 计算 LFR 输出帧数
	outFrames := (numFrames-lfrM)/lfrN + 1
	outDim := inputDim * lfrM

	output := make([][]float32, outFrames)

	for i := 0; i < outFrames; i++ {
		output[i] = make([]float32, outDim)
		startFrame := i * lfrN

		// 拼接 M 帧
		for j := 0; j < lfrM; j++ {
			srcIdx := startFrame + j
			// 目标位置: j * 80 到 (j+1) * 80
			destPos := j * inputDim
			copy(output[i][destPos:], inputs[srcIdx])
		}
	}

	return output, outFrames
}
