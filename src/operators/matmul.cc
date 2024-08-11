#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A=inputs[0];
        auto B=inputs[1];
        auto A_dim = A->getDims();
        auto B_dim = B->getDims();
        int rank=A->getRank();
        auto output_dim = A_dim;
        for(int i=0;i<rank-2;++i)
            output_dim[i]=std::max(A_dim[i],B_dim[i]);
        if(transA)
            output_dim[rank-2]=A_dim[rank-1];
        else
            output_dim[rank-2]=A_dim[rank-2];

        if(transB)
            output_dim[rank-1]=B_dim[rank-2];
        else
            output_dim[rank-1]=B_dim[rank-1];
        vector<Shape>ans={output_dim};
        return ans;
    }

} // namespace infini