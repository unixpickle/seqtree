package seqtree

import (
	"math"
	"math/rand"
	"testing"
)

func TestSoftmaxLoss(t *testing.T) {
	for i := 0; i < 2560; i++ {
		size := i/10 + 2
		vec := make([]float32, size)
		target := make([]float32, len(vec))
		for i := range vec {
			vec[i] = float32(rand.NormFloat64())
			target[i] = float32(rand.NormFloat64())
		}
		if i%2 == 0 {
			// Test extreme cases.
			for j := range vec {
				if rand.Intn(2) == 0 {
					vec[j] *= float32(rand.NormFloat64() * 10)
				}
			}
		}
		actual := Softmax{}.Loss(vec, target)
		expected := float32(0)
		log := Softmax{}.logSoftmax(vec)
		for i, l := range log {
			expected -= l * target[i]
		}
		if math.Abs(float64(actual-expected)) > math.Abs(float64(expected))*1e-4 {
			t.Errorf("expected %f but got %f", expected, actual)
		}
	}
}

func BenchmarkSoftmaxLoss(b *testing.B) {
	b.Run("Size2", func(b *testing.B) {
		params := []float32{-0.5, 0.7}
		targets := []float32{0.2, 0.3}
		for i := 0; i < b.N; i++ {
			Softmax{}.Loss(params, targets)
		}
	})
	b.Run("Size10", func(b *testing.B) {
		params := make([]float32, 10)
		targets := make([]float32, 10)
		for i := range params {
			params[i] = float32(rand.NormFloat64())
			targets[i] = float32(rand.NormFloat64())
		}
		for i := 0; i < b.N; i++ {
			Softmax{}.Loss(params, targets)
		}
	})
}

func BenchmarkSoftmaxLossGrad(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Softmax{}.LossGrad([]float32{1.5, -0.3}, []float32{1.0, 0})
	}
}
