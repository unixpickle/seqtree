package seqtree

import (
	"math"
	"math/rand"
	"testing"
)

func TestSoftmaxLoss(t *testing.T) {
	for i := 0; i < 2560; i++ {
		size := i/10 + 2
		vec := make([]float64, size)
		target := make([]float64, len(vec))
		for i := range vec {
			vec[i] = float64(rand.NormFloat64())
			target[i] = float64(rand.NormFloat64())
		}
		if i%2 == 0 {
			// Test extreme cases.
			for j := range vec {
				if rand.Intn(2) == 0 {
					vec[j] *= float64(rand.NormFloat64() * 10)
				}
			}
		}
		actual := SoftmaxLoss(vec, target)
		expected := float64(0)
		log := logSoftmax(vec)
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
		params := []float64{-0.5, 0.7}
		targets := []float64{0.2, 0.3}
		for i := 0; i < b.N; i++ {
			SoftmaxLoss(params, targets)
		}
	})
	b.Run("Size10", func(b *testing.B) {
		params := make([]float64, 10)
		targets := make([]float64, 10)
		for i := range params {
			params[i] = float64(rand.NormFloat64())
			targets[i] = float64(rand.NormFloat64())
		}
		for i := 0; i < b.N; i++ {
			SoftmaxLoss(params, targets)
		}
	})
}

func BenchmarkSoftmaxLossGrad(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SoftmaxLossGrad([]float64{1.5, -0.3}, []float64{1.0, 0})
	}
}

func BenchmarkSoftmaxLossDelta(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SoftmaxLossDelta([]float64{1.5, -0.3}, []float64{1.0, 0}, []float64{0.5, -0.2}, 0.1)
	}
}

func BenchmarkSoftmaxLossKL(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SoftmaxLossKL([]float64{1.5, -0.3}, []float64{0.5, -0.2}, 0.1)
	}
}
