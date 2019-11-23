package seqtree

import (
	"math"
	"math/rand"
)

// SampleSoftmax samples a softmax distribution from
// logits.
func SampleSoftmax(outputs []float32) int {
	logProbs := logSoftmax(outputs)
	p := rand.Float32()
	for i, x := range logProbs {
		p -= float32(math.Exp(float64(x)))
		if p <= 0 {
			return i
		}
	}
	return len(outputs) - 1
}

// SoftmaxLoss computes the loss function given output
// logits and target probabilities.
func SoftmaxLoss(outputs, targets []float32) float32 {
	if len(outputs) == 2 {
		// Fast case with one fewer exponential.
		var normer float32
		if outputs[0] > outputs[1] {
			normer = outputs[0] + float32(math.Log(1+math.Exp(float64(outputs[1]-outputs[0]))))
		} else {
			normer = outputs[1] + float32(math.Log(1+math.Exp(float64(outputs[0]-outputs[1]))))
		}
		return targets[0]*(normer-outputs[0]) + targets[1]*(normer-outputs[1])
	}

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

	subtractor := max + float32(math.Log(float64(sumOfExp)))

	var loss float32
	for i, x := range outputs {
		loss += (subtractor - x) * targets[i]
	}
	return loss
}

// SoftmaxLossGrad computes the gradient of SoftmaxLoss
// with respect to the outputs.
func SoftmaxLossGrad(outputs, targets []float32) []float32 {
	var targetSum float32
	for _, x := range targets {
		targetSum += x
	}

	maxOutput := outputs[0]
	for _, x := range outputs[1:] {
		if x > maxOutput {
			maxOutput = x
		}
	}

	grad := make([]float32, len(outputs))
	for i, x := range outputs {
		grad[i] = float32(math.Exp(float64(x - maxOutput)))
	}
	var gradSum float32
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
func SoftmaxLossNaturalGrad(outputs, targets []float32) []float32 {
	prob := float32(math.Exp(float64(-SoftmaxLoss(outputs, targets))))

	// Add a small epsilon to prevent divide-by-zero.
	prob += 1e-4

	baseGrad := 1.0 / (float32(len(outputs)) * prob)
	naturalGrad := make([]float32, len(outputs))
	for i, x := range targets {
		naturalGrad[i] = baseGrad * (1 - x*float32(len(outputs)))
	}
	return naturalGrad
}

// SoftmaxLossHessian computes the Hessian matrix for the
// softmax loss function.
func SoftmaxLossHessian(outputs, targets []float32) *SquareMatrix {
	// TODO: use an exact formula here instead of finite
	// differences.
	const Epsilon = 1e-4
	result := &SquareMatrix{
		Dim:  len(outputs),
		Data: make([]float32, 0, len(outputs)*len(outputs)),
	}
	newOutputs := append([]float32{}, outputs...)
	for i, x := range outputs {
		newOutputs[i] = x - Epsilon
		grad1 := SoftmaxLossGrad(newOutputs, targets)
		newOutputs[i] = x + Epsilon
		grad2 := SoftmaxLossGrad(newOutputs, targets)
		newOutputs[i] = x
		for i, g1 := range grad1 {
			result.Data = append(result.Data, (grad2[i]-g1)/(Epsilon*2))
		}
	}
	return result
}

// SoftmaxLossKL computes
//
//     KL(softmax(outputs+deltas*stepSize)||softmax(outputs))
//
// This can be used to bound the KL of updates.
func SoftmaxLossKL(outputs, deltas []float32, stepSize float32) float32 {
	oldLogProbs := logSoftmax(outputs)
	newLogProbs := logSoftmax(addDelta(outputs, deltas, stepSize))
	var res float32
	for i, x := range newLogProbs {
		res += float32(math.Exp(float64(x)) * float64(x-oldLogProbs[i]))
	}
	return res
}

// SoftmaxLossDelta computes how much the loss changes due
// to adding deltas*stepSize to outputs.
func SoftmaxLossDelta(outputs, targets, deltas []float32, stepSize float32) float32 {
	loss1 := SoftmaxLoss(outputs, targets)
	loss2 := SoftmaxLoss(addDelta(outputs, deltas, stepSize), targets)
	return loss2 - loss1
}

func logSoftmax(logits []float32) []float32 {
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

	subtractor := max + float32(math.Log(float64(sumOfExp)))

	res := make([]float32, len(logits))
	for i, x := range logits {
		res[i] = x - subtractor
	}

	return res
}

func addDelta(v1, v2 []float32, scale float32) []float32 {
	res := make([]float32, len(v1))
	for i, x := range v1 {
		res[i] = x + v2[i]*scale
	}
	return res
}
