package seqtree

import (
	"fmt"
	"math"
	"runtime"
	"testing"
)

func TestOptimalStep(t *testing.T) {
	for _, baseFeatures := range []int{2, 10} {
		name := fmt.Sprintf("Features%d", baseFeatures)
		t.Run(name, func(t *testing.T) {
			for i := 0; i < 10; i++ {
				m := generateTestModel(baseFeatures)
				b := &Builder{
					Depth:           3,
					MinSplitSamples: 10,
					Horizons:        []int{0, 1, 2},
				}
				ts := TimestepSamples(generateTestSequences(m))
				tree := b.Build(ts)

				actual := OptimalStep(ts, tree, Softmax{}, 40.0, 100)
				actualLoss := AvgLossDelta(ts, tree, Softmax{}, actual)
				expected := bruteForceOptimalStep(ts, tree, 5, 0, 40.0)
				expectedLoss := AvgLossDelta(ts, tree, Softmax{}, expected)
				if math.Abs(float64(actualLoss-expectedLoss)) > 1e-4 {
					t.Errorf("actual step is %f (loss=%f), expected is %f (loss=%f)",
						actual, AvgLossDelta(ts, tree, Softmax{}, actual),
						expected, AvgLossDelta(ts, tree, Softmax{}, expected))
				}
			}
		})
	}
}

func BenchmarkOptimalStep(b *testing.B) {
	oldCount := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(oldCount)

	for _, baseFeatures := range []int{2, 10} {
		name := fmt.Sprintf("Features%d", baseFeatures)
		b.Run(name, func(b *testing.B) {
			for i := 0; i < 10; i++ {
				b.StopTimer()
				m := generateTestModel(baseFeatures)
				builder := &Builder{
					Depth:           3,
					MinSplitSamples: 10,
					Horizons:        []int{0, 1, 2},
				}
				ts := TimestepSamples(generateTestSequences(m))
				tree := builder.Build(ts)
				b.StartTimer()

				for j := 0; j < b.N; j++ {
					OptimalStep(ts, tree, Softmax{}, 40.0, 100)
				}
			}
		})
	}
}

func bruteForceOptimalStep(ts []*TimestepSample, t *Tree, stepsLeft int, min, max float32) float32 {
	var minimum float32
	var minStep float32
	for i := 0; i <= 1000; i++ {
		s := (max-min)*float32(i)/1000 + min
		delta := AvgLossDelta(ts, t, Softmax{}, s)
		if delta < minimum {
			minimum = delta
			minStep = s
		}
	}
	if stepsLeft > 0 {
		return bruteForceOptimalStep(ts, t, stepsLeft-1, minStep-(max-min)/100,
			minStep+(max-min)/100)
	}
	return minStep
}
