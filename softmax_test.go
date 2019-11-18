package seqtree

import "testing"

func BenchmarkSoftmaxLossGrad(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SoftmaxLossGrad([]float32{1.5, -0.3}, []float32{1.0, 0})
	}
}
