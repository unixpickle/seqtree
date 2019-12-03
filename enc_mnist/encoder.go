package main

import (
	"log"
	"math/rand"
	"runtime"
	"sync"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/seqtree"
)

const (
	ImageSize = 28
	BatchSize = 5000

	EncodingDim1    = 40
	EncodingDim2    = 20
	EncodingOptions = 16
)

func TrainEncoder(e *Encoder, ds, testDs mnist.DataSet) {
	if len(e.Layer1.Stages) < EncodingDim1 {
		trainEncoderLayer1(e, ds, testDs)
	}
	if len(e.Layer2.Stages) < EncodingDim2 {
		trainEncoderLayer2(e, ds, testDs)
	}
}

func trainEncoderLayer1(e *Encoder, ds, testDs mnist.DataSet) {
	sampleVecs := func(ds mnist.DataSet) [][]float32 {
		var vecs [][]float32
		for i := 0; i < BatchSize; i++ {
			vecs = append(vecs, encodeSigmoid(ds.Samples[rand.Intn(len(ds.Samples))].Intensities))
		}
		return vecs
	}
	for len(e.Layer1.Stages) < EncodingDim1 {
		vecs := sampleVecs(ds)
		testVecs := sampleVecs(testDs)
		loss := evaluateLoss(e.Layer1, vecs)
		testLoss := evaluateLoss(e.Layer1, testVecs)
		e.Layer1.AddStage(&seqtree.KMeans{
			MaxIterations: 100,
			NumClusters:   EncodingOptions,
		}, vecs)
		log.Printf("layer 1: step %d: loss=%f test=%f", len(e.Layer1.Stages)-1, loss, testLoss)
	}
}

func trainEncoderLayer2(e *Encoder, ds, testDs mnist.DataSet) {
	sampleVecs := func(ds mnist.DataSet) [][]float32 {
		var vecs [][]float32
		for i := 0; i < BatchSize; i++ {
			enc := e.EncodeLayer1(ds.Samples[rand.Intn(len(ds.Samples))].Intensities)
			vecs = append(vecs, encodeOneHot(enc))
		}
		return vecs
	}
	for len(e.Layer2.Stages) < EncodingDim2 {
		vecs := sampleVecs(ds)
		testVecs := sampleVecs(testDs)
		loss := evaluateLoss(e.Layer2, vecs)
		testLoss := evaluateLoss(e.Layer2, testVecs)
		e.Layer2.AddStage(&seqtree.KMeans{
			MaxIterations: 100,
			NumClusters:   EncodingOptions,
		}, vecs)
		log.Printf("layer 2: step %d: loss=%f test=%f", len(e.Layer2.Stages)-1, loss, testLoss)
	}
}

func evaluateLoss(e *seqtree.ClusterEncoder, vecs [][]float32) float32 {
	var res float32
	for _, v := range vecs {
		dec := make([]float32, len(v))
		if len(e.Stages) > 0 {
			dec = e.Decode(e.Encode(v))
		}
		res += e.Loss.Loss(dec, v)
	}
	return res / float32(len(vecs))
}

type Encoder struct {
	Layer1 *seqtree.ClusterEncoder
	Layer2 *seqtree.ClusterEncoder
}

func NewEncoder() *Encoder {
	var sizes []int
	for i := 0; i < EncodingDim1; i++ {
		sizes = append(sizes, EncodingOptions)
	}
	return &Encoder{
		Layer1: &seqtree.ClusterEncoder{
			Loss: seqtree.Sigmoid{},
		},
		Layer2: &seqtree.ClusterEncoder{
			Loss: seqtree.MultiSoftmax{Sizes: sizes},
		},
	}
}

func (e *Encoder) NeedsTraining() bool {
	return len(e.Layer1.Stages) < EncodingDim1 ||
		len(e.Layer2.Stages) < EncodingDim2
}

func (e *Encoder) EncodeLayer1(image []float64) []int {
	return e.Layer1.Encode(encodeSigmoid(image))
}

func (e *Encoder) EncodeLayer2(intSeq []int) []int {
	return e.Layer2.Encode(encodeOneHot(intSeq))
}

func (e *Encoder) Encode(image []float64) []int {
	return e.EncodeLayer2(e.EncodeLayer1(image))
}

func (e *Encoder) EncodeBatch(ds mnist.DataSet, n int) [][]int {
	perm := rand.Perm(len(ds.Samples))[:n]
	res := make([][]int, n)
	numProcs := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := i; j < n; j += numProcs {
				res[j] = e.Encode(ds.Samples[perm[j]].Intensities)
			}
		}(i)
	}
	wg.Wait()
	return res
}

func (e *Encoder) DecodeLayer1(seq []int) []float64 {
	rawRes := e.Layer1.Decode(seq)
	var res []float64
	for _, x := range rawRes {
		res = append(res, float64(x))
	}
	return res
}

func (e *Encoder) DecodeLayer2(seq []int) []int {
	params := e.Layer2.Decode(seq)
	res := make([]int, EncodingDim1)
	for i := range res {
		maxIdx := 0
		maxValue := params[i*EncodingOptions]
		for j, x := range params[i*EncodingOptions : (i+1)*EncodingOptions] {
			if x > maxValue {
				maxValue = x
				maxIdx = j
			}
		}
		res[i] = maxIdx
	}
	return res
}

func (e *Encoder) Decode(seq []int) []float64 {
	return e.DecodeLayer1(e.DecodeLayer2(seq))
}

func encodeSigmoid(seq []float64) []float32 {
	targets := make([]float32, len(seq))
	for i, x := range seq {
		if x >= 0.5 {
			targets[i] = 1
		}
	}
	return targets
}

func encodeOneHot(seq []int) []float32 {
	res := make([]float32, len(seq)*EncodingOptions)
	for i, x := range seq {
		res[i*EncodingOptions+x] = 1
	}
	return res
}
