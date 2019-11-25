package seqtree

type kahanSum struct {
	sum          []float64
	compensation []float64
}

func newKahanSum(dim int) *kahanSum {
	return &kahanSum{
		sum:          make([]float64, dim),
		compensation: make([]float64, dim),
	}
}

func (k *kahanSum) Add(addition []float64) {
	for i, n := range addition {
		n -= k.compensation[i]
		sum := k.sum[i] + n
		k.compensation[i] = (sum - k.sum[i]) - n
		k.sum[i] = sum
	}
}

func (k *kahanSum) Sum() []float64 {
	return k.sum
}
