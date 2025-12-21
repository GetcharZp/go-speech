package whisper

import (
	"github.com/up-zero/gotool/mediautil"
	"math"
	"sync"
)

const (
	// sampleRate 采样率
	sampleRate = 16000
	// channels 声道数
	channels = 1
	// bitsPerSample 采样位数
	bitsPerSample = 16

	nFFT    = 512
	winLen  = 400
	hopLen  = 160
	nMel    = 80
	maxSmpl = 480000
	nFr     = 3000
)

var (
	window     []float32
	melFilters [][]float32
	specOnce   sync.Once
)

// extractFeatures 特征处理
func (e *Engine) extractFeatures(samples []float32) ([]float32, error) {
	specOnce.Do(func() {
		window = mediautil.HannWindow(winLen)
		melFilters = mediautil.MelFilters(sampleRate, nFFT, nMel, 0, 0)
	})

	padded := make([]float32, maxSmpl)
	copy(padded, samples)
	ref := padReflect(padded, nFFT/2)
	flat := make([]float32, nMel*nFr)
	tempSpec := make([][]float32, nFr)
	maxV := float32(-math.MaxFloat32)
	fftBuffer := make([]complex128, nFFT)

	for i := 0; i < nFr; i++ {
		start := i * hopLen
		for j := 0; j < nFFT; j++ {
			if j < winLen {
				val := ref[start+j] * window[j]
				fftBuffer[j] = complex(float64(val), 0)
			}
		}
		spectrum := mediautil.FFT(fftBuffer)

		// 梅尔频谱
		tempSpec[i] = make([]float32, nMel)
		for k := 0; k < nMel; k++ {
			sum := 0.0
			for j := 0; j < nFFT/2+1; j++ {
				if w := melFilters[k][j]; w > 0 {
					r := real(spectrum[j])
					im := imag(spectrum[j])
					power := r*r + im*im
					sum += power * float64(w)
				}
			}
			val := float32(math.Log10(max(sum, 1e-10)))
			tempSpec[i][k] = val
			if val > maxV {
				maxV = val
			}
		}
	}

	for i := 0; i < nFr; i++ {
		for k := 0; k < nMel; k++ {
			v := (max(tempSpec[i][k], maxV-8.0) + 4.0) / 4.0
			flat[k*nFr+i] = v
		}
	}
	return flat, nil
}

func padReflect(s []float32, p int) []float32 {
	n := len(s)
	res := make([]float32, n+2*p)
	for i := 0; i < p; i++ {
		res[i] = s[p-i]
		res[n+p+i] = s[n-1-i]
	}
	copy(res[p:], s)
	return res
}
