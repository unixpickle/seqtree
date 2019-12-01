package main

import (
	"image"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"
)

const Batch = 1000000

func main() {
	data := DatasetBoolImgs(mnist.LoadTrainingDataSet())
	testData := DatasetBoolImgs(mnist.LoadTestingDataSet())

	seqModel := NewSequenceModel()
	seqModel.Model.Load("model.json")
	for i := 0; true; i++ {
		testSeqs := seqModel.Timesteps(testData, Batch)
		testLoss := seqModel.MeanLoss(testSeqs)

		trainSeqs := seqModel.Timesteps(data, Batch)
		validSeqs := seqModel.Timesteps(data, Batch)
		loss, delta := seqModel.AddTree(trainSeqs, validSeqs)

		log.Printf("tree %d: loss=%f delta=%f test=%f", len(seqModel.Model.Trees)-1, loss, -delta,
			testLoss)
		seqModel.Model.Save("model.json")
		GenerateSamples(seqModel)
	}
}

func CollectSamples(ds mnist.DataSet, n int) []mnist.Sample {
	res := make([]mnist.Sample, n)
	for i := 0; i < n; i++ {
		res[i] = ds.Samples[rand.Intn(len(ds.Samples))]
	}
	return res
}

func GenerateSamples(s *SequenceModel) {
	img := image.NewGray(image.Rect(0, 0, ImageSize*4, ImageSize*4))
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			pixels := s.Sample()
			for i := 0; i < ImageSize; i++ {
				for j := 0; j < ImageSize; j++ {
					pixel := 0
					if pixels[i*ImageSize+j] {
						pixel = 255
					}
					img.SetGray(row*ImageSize+j, col*ImageSize+i, color.Gray{Y: uint8(pixel)})
				}
			}
		}
	}
	w, err := os.Create("sample.png")
	essentials.Must(err)
	defer w.Close()
	essentials.Must(png.Encode(w, img))
}

func DatasetBoolImgs(ds mnist.DataSet) []BoolImg {
	res := make([]BoolImg, len(ds.Samples))
	for i, s := range ds.Samples {
		res[i] = NewBoolImgSample(s)
	}
	return res
}
