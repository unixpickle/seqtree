package seqtree

type kahanSum struct {
	sum          []float32
	compensation []float32
}

func newKahanSum(dim int) *kahanSum {
	return &kahanSum{
		sum:          make([]float32, dim),
		compensation: make([]float32, dim),
	}
}

func (k *kahanSum) Add(addition []float32) {
	for i, n := range addition {
		n -= k.compensation[i]
		sum := k.sum[i] + n
		k.compensation[i] = (sum - k.sum[i]) - n
		k.sum[i] = sum
	}
}

func (k *kahanSum) Sum() []float32 {
	return k.sum
}
