package seqtree

import (
	"math"
	"math/rand"
	"testing"
)

func TestHessianInverse(t *testing.T) {
	const MatrixSize = 16

	// Generate a symmetric positive definite Hessian.
	rawVectors := make([][]float32, MatrixSize)
	for i := range rawVectors {
		rawVectors[i] = make([]float32, MatrixSize)
		for j := range rawVectors[i] {
			rawVectors[i][j] = float32(rand.NormFloat64())
		}
	}
	hess := &Hessian{
		Dim:  16,
		Data: make([]float32, MatrixSize*MatrixSize),
	}
	for i := 0; i < hess.Dim; i++ {
		for j := 0; j < hess.Dim; j++ {
			hess.Data[i+j*hess.Dim] = vectorDot(rawVectors[i], rawVectors[j])
		}
	}

	targetVector := make([]float32, hess.Dim)
	for i := range targetVector {
		targetVector[i] = float32(rand.NormFloat64())
	}

	solution := hess.ApplyInverse(targetVector)
	residual := vectorDifference(hess.Apply(solution), targetVector)
	residualMag := math.Sqrt(float64(vectorNormSquared(residual)))
	if residualMag > 1e-4 {
		t.Errorf("residual error of %f exceeds threshold", residualMag)
	}

	zeroTarget := make([]float32, hess.Dim)
	solution = hess.ApplyInverse(zeroTarget)
	for _, x := range solution {
		if x != 0 {
			t.Errorf("expected 0 but got %f", x)
		}
	}
}
