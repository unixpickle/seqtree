package seqtree

import (
	"math"
	"math/rand"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

// SampleSoftmax samples a softmax distribution from
// logits.
func SampleSoftmax(outputs []float32) int {
	probs := anyvec32.MakeVectorData(outputs)
	anyvec.LogSoftmax(probs, len(outputs))
	anyvec.Exp(probs)
	vec := probs.Data().([]float32)
	p := rand.Float32()
	for i, x := range vec {
		p -= x
		if p <= 0 {
			return i
		}
	}
	return len(vec) - 1
}

// SoftmaxLoss computes the loss function given output
// logits and target probabilities.
func SoftmaxLoss(outputs, targets []float32) float32 {
	probs := logSoftmax(outputs)
	var loss float32
	for i, x := range targets {
		loss -= x * probs[i]
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
