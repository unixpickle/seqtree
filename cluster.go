package seqtree

import (
	"encoding/json"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"reflect"
	"runtime"
	"sync"

	"github.com/pkg/errors"
)

type ClusterEncoder struct {
	Stages  []*Clusters
	Weights []float32
	Loss    GradLossFunc `json:"-"`
}

// Save saves the model to a JSON file.
func (c *ClusterEncoder) Save(path string) error {
	data, err := json.Marshal(c)
	if err != nil {
		return errors.Wrap(err, "save encoder")
	}
	if err := ioutil.WriteFile(path, data, 0755); err != nil {
		return errors.Wrap(err, "save encoder")
	}
	return nil
}

// Load loads the model from a JSON file.
// Does not fail with an error if the file does not exist.
func (c *ClusterEncoder) Load(path string) error {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return errors.Wrap(err, "load encoder")
	}
	if err := json.Unmarshal(data, c); err != nil {
		return errors.Wrap(err, "load encoder")
	}
	return nil
}

func (c *ClusterEncoder) AddStage(k *KMeans, data [][]float32, errorRate float32) {
	zeroOutput := make([]float32, len(data[0]))
	prevOutputs := make([][]float32, len(data))
	grads := make([][]float32, len(data))

	var lock sync.Mutex
	originalLoss := newKahanSum(1)

	var wg sync.WaitGroup
	numProcs := runtime.GOMAXPROCS(0)
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			localSum := newKahanSum(1)
			for j := i; j < len(data); j += numProcs {
				_, prevOutput := c.encodeWithOutput(zeroOutput, data[j])
				grads[j] = c.Loss.LossGrad(prevOutput, data[j])
				prevOutputs[j] = prevOutput
				localSum.Add([]float32{c.Loss.Loss(prevOutput, data[j])})
			}
			lock.Lock()
			originalLoss.Add(localSum.Sum())
			lock.Unlock()
		}(i)
	}
	wg.Wait()

	centers := k.Cluster(grads)
	clusters := &Clusters{
		Centers: centers,
		Deltas:  make([][]float32, len(centers)),
	}

	clusterIdxs := make([]int, len(prevOutputs))
	for i, x := range grads {
		clusterIdxs[i], _ = clusters.Find(x)
	}

	newLoss := newKahanSum(1)
	for i, center := range centers {
		delta := append([]float32{}, center...)
		for i := range delta {
			delta[i] *= -1
		}

		var newData, targets, otherData, otherTargets [][]float32
		for j, c := range clusterIdxs {
			if c == i {
				newData = append(newData, prevOutputs[j])
				targets = append(targets, data[j])
			} else {
				otherData = append(otherData, prevOutputs[j])
				otherTargets = append(otherTargets, data[j])
			}
		}

		otherWeight := errorRate * float32(len(newData)) / float32(len(otherData))

		if m, ok := c.Loss.(*MultiSoftmax); ok {
			start := 0
			for _, s := range m.Sizes {
				scaleOptimalStepCluster(newData, targets, otherData, otherTargets, delta, Softmax{},
					100.0, otherWeight, 50, start, s)
				start += s
			}
		} else if _, ok := c.Loss.(Sigmoid); ok {
			for i := range delta {
				scaleOptimalStepCluster(newData, targets, otherData, otherTargets, delta, c.Loss,
					100.0, otherWeight, 50, i, 1)
			}
		} else {
			scaleOptimalStepCluster(newData, targets, otherData, otherTargets, delta, c.Loss, 100.0,
				otherWeight, 50, 0, 0)
		}

		clusters.Deltas[i] = delta
		for j, x := range data {
			t := targets[j]
			newLoss.Add([]float32{c.Loss.Loss(addDelta(x, delta, 1), t)})
		}
	}

	weight := (originalLoss.Sum()[0] - newLoss.Sum()[0]) / float32(len(data))

	c.Stages = append(c.Stages, clusters)
	c.Weights = append(c.Weights, weight)
}

func (c *ClusterEncoder) Encode(targets []float32) []int {
	res, _ := c.encodeWithOutput(nil, targets)
	return res
}

func (c *ClusterEncoder) Decode(code []int) []float32 {
	res := newKahanSum(len(c.Stages[0].Deltas[0]))
	for i, x := range code {
		res.Add(c.Stages[i].Deltas[x])
	}
	return res.Sum()
}

func (c *ClusterEncoder) encodeWithOutput(outputs, targets []float32) ([]int, []float32) {
	if outputs == nil {
		outputs = make([]float32, len(targets))
	}
	var result []int
	for _, stage := range c.Stages {
		lossGrad := c.Loss.LossGrad(outputs, targets)
		idx, _ := stage.Find(lossGrad)
		outputs = addDelta(outputs, stage.Deltas[idx], 1.0)
		result = append(result, idx)
	}
	return result, outputs
}

type Clusters struct {
	Deltas  [][]float32
	Centers [][]float32
}

// Find finds the closest center index to v and also
// returns the squared distance.
func (c *Clusters) Find(v []float32) (int, float32) {
	minDist := float32(math.Inf(1))
	minIdx := 0
	for i, center := range c.Centers {
		dist := vectorNormSquared(vectorDifference(v, center))
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}
	return minIdx, minDist
}

// KMeans builds clusters.
type KMeans struct {
	NumClusters   int
	MaxIterations int
}

// Cluster clusters the data points into centers.
func (k *KMeans) Cluster(data [][]float32) [][]float32 {
	result := k.initialize(data)
	lastCenters := result
	for i := 0; i < k.MaxIterations; i++ {
		result = k.iterate(data, result)
		if reflect.DeepEqual(result, lastCenters) {
			break
		}
		lastCenters = result
	}
	return result
}

func (k *KMeans) initialize(data [][]float32) [][]float32 {
	var result [][]float32

	// Random initial center.
	result = append(result, data[rand.Intn(len(data))])

	// Use k-means++ to sample remaining centers.
	for len(result) < k.NumClusters {
		var sqDists []float32
		totalDist := newKahanSum(1)
		for _, x := range data {
			_, dist := (&Clusters{Centers: result}).Find(x)
			sqDists = append(sqDists, dist)
			totalDist.Add([]float32{dist})
		}

		p := rand.Float32() * totalDist.Sum()[0]

		totalDist = newKahanSum(1)
		for i, dist := range sqDists {
			totalDist.Add([]float32{dist})
			if totalDist.Sum()[0] >= p {
				result = append(result, data[i])
				break
			}
		}
		if totalDist.Sum()[0] < p {
			// Rare edge case, likely due to rounding.
			result = append(result, data[len(data)-1])
		}
	}

	return result
}

func (k *KMeans) iterate(data, centers [][]float32) [][]float32 {
	dim := len(centers[0])

	var lock sync.Mutex
	counts := make([]int, len(centers))
	sums := make([]*kahanSum, len(centers))
	for i := range sums {
		sums[i] = newKahanSum(dim)
	}

	c := &Clusters{Centers: centers}

	numProcs := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			localCounts := make([]int, len(centers))
			localSums := make([]*kahanSum, len(centers))
			for i := range localSums {
				localSums[i] = newKahanSum(dim)
			}
			for j := i; j < len(data); j += numProcs {
				x := data[j]
				cluster, _ := c.Find(x)
				localSums[cluster].Add(x)
				localCounts[cluster] += 1
			}
			lock.Lock()
			for i, ls := range localSums {
				sums[i].Add(ls.Sum())
			}
			for i, x := range localCounts {
				counts[i] += x
			}
			lock.Unlock()
		}(i)
	}
	wg.Wait()

	res := make([][]float32, len(sums))
	for i, s := range sums {
		res[i] = make([]float32, dim)
		for j, x := range s.Sum() {
			res[i][j] = x / float32(counts[i])
		}
	}

	return res
}
