package seqtree

import (
	"math"
	"math/rand"
)

// SampleSoftmax samples a softmax distribution from
// logits.
func SampleSoftmax(outputs []float64) int {
	logProbs := logSoftmax(outputs)
	p := rand.Float64()
	for i, x := range logProbs {
		p -= float64(math.Exp(float64(x)))
		if p <= 0 {
			return i
		}
	}
	return len(outputs) - 1
}

// SoftmaxLoss computes the loss function given output
// logits and target probabilities.
func SoftmaxLoss(outputs, targets []float64) float64 {
	// Code duplication from logSoftmax() prevents
	// allocations and can speedup computation
	// significantly.

	max := outputs[0]
	for _, x := range outputs[1:] {
		if x > max {
			max = x
		}
	}

	var sumOfExp float64
	for _, x := range outputs {
		sumOfExp += math.Exp(float64(x - max))
	}

	subtractor := max + float64(math.Log(float64(sumOfExp)))

	var loss float64
	for i, x := range outputs {
		loss += (subtractor - x) * targets[i]
	}
	return loss
}

// SoftmaxLossGrad computes the gradient of SoftmaxLoss
// with respect to the outputs.
func SoftmaxLossGrad(outputs, targets []float64) []float64 {
	var targetSum float64
	for _, x := range targets {
		targetSum += x
	}

	maxOutput := outputs[0]
	for _, x := range outputs[1:] {
		if x > maxOutput {
			maxOutput = x
		}
	}

	grad := make([]float64, len(outputs))
	for i, x := range outputs {
		grad[i] = float64(math.Exp(float64(x - maxOutput)))
	}
	var gradSum float64
	for _, x := range grad {
		gradSum += x
	}
	div := 1.0 / gradSum
	for i, x := range grad {
		grad[i] = targetSum*x*div - targets[i]
	}
	return grad
}

// SoftmaxLossNaturalGrad is like SoftmaxLossGrad, except
// the natural functional gradient is used.
func SoftmaxLossNaturalGrad(outputs, targets []float64) []float64 {
	prob := float64(math.Exp(float64(-SoftmaxLoss(outputs, targets))))

	// Add a small epsilon to prevent divide-by-zero.
	prob += 1e-4

	baseGrad := 1.0 / (float64(len(outputs)) * prob)
	naturalGrad := make([]float64, len(outputs))
	for i, x := range targets {
		naturalGrad[i] = baseGrad * (1 - x*float64(len(outputs)))
	}
	return naturalGrad
}

// SoftmaxLossKL computes
//
//     KL(softmax(outputs+deltas*stepSize)||softmax(outputs))
//
// This can be used to bound the KL of updates.
func SoftmaxLossKL(outputs, deltas []float64, stepSize float64) float64 {
	oldLogProbs := logSoftmax(outputs)
	newLogProbs := logSoftmax(addDelta(outputs, deltas, stepSize))
	var res float64
	for i, x := range newLogProbs {
		res += float64(math.Exp(float64(x)) * float64(x-oldLogProbs[i]))
	}
	return res
}

// SoftmaxLossDelta computes how much the loss changes due
// to adding deltas*stepSize to outputs.
func SoftmaxLossDelta(outputs, targets, deltas []float64, stepSize float64) float64 {
	loss1 := SoftmaxLoss(outputs, targets)
	loss2 := SoftmaxLoss(addDelta(outputs, deltas, stepSize), targets)
	return loss2 - loss1
}

func logSoftmax(logits []float64) []float64 {
	max := logits[0]
	for _, x := range logits[1:] {
		if x > max {
			max = x
		}
	}

	var sumOfExp float64
	for _, x := range logits {
		sumOfExp += math.Exp(float64(x - max))
	}

	subtractor := max + float64(math.Log(float64(sumOfExp)))

	res := make([]float64, len(logits))
	for i, x := range logits {
		res[i] = x - subtractor
	}

	return res
}

func addDelta(v1, v2 []float64, scale float64) []float64 {
	res := make([]float64, len(v1))
	for i, x := range v1 {
		res[i] = x + v2[i]*scale
	}
	return res
}
