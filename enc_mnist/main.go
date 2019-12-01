package main

import (
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"
)

func main() {
	dataset := mnist.LoadTrainingDataSet()
	testDataset := mnist.LoadTestingDataSet()

	encoder := NewEncoder()
	encoder.Layer1.Load("encoder1.json")
	encoder.Layer2.Load("encoder2.json")
	if encoder.NeedsTraining() {
		log.Println("Training encoder...")
		TrainEncoder(encoder, dataset)
		encoder.Layer1.Save("encoder1.json")
		encoder.Layer2.Save("encoder2.json")
	}
	log.Println("Saving encoder reconstructions...")
	GenerateReconstructions(testDataset, encoder)

	seqModel := NewSequenceModel()
	seqModel.Load("sequence_model.json")
	log.Println("Training sequence model...")
	seqs := encoder.EncodeBatch(dataset, len(dataset.Samples))
	testSeqs := encoder.EncodeBatch(testDataset, len(testDataset.Samples))
	for i := 0; true; i++ {
		trainStart := (i * 10000) % len(dataset.Samples)
		testLoss := seqModel.MeanLoss(testSeqs)
		loss, delta := seqModel.AddTree(seqs[trainStart : trainStart+10000])
		log.Printf("tree %d: loss=%f delta=%f test=%f", seqModel.NumTrees()-1, loss, -delta,
			testLoss)
		seqModel.Save("sequence_model.json")
		GenerateSamples(encoder, seqModel)
	}
}

func GenerateSamples(e *Encoder, s *SequenceModel) {
	img := image.NewGray(image.Rect(0, 0, ImageSize*4, ImageSize*4))
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			pixels := e.Decode(s.Sample())
			for i := 0; i < ImageSize; i++ {
				for j := 0; j < ImageSize; j++ {
					pixel := 255 / (1 + math.Exp(-pixels[i*ImageSize+j]))
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

func GenerateReconstructions(ds mnist.DataSet, e *Encoder) {
	mnist.SaveReconstructionGrid("recon.png", func(x []float64) []float64 {
		vec := e.Decode(e.Encode(x))
		for i, x := range vec {
			vec[i] = 1 / (1 + math.Exp(-x))
		}
		return vec
	}, ds, 10, 4)
}
