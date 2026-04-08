struct Reservoir{

    uint M;             // Number of samples. Real number equals generatedSamples * M
    float targetFunc; // Target Pdf of current sample
    float weightSum;    // Weight of the reservoir after finalizing (W in paper). Used as weightSum internaly
    uint pad;           // Pad for fitting
    
    //Empty init
    __init(){
        this = {};
        M = 0;
        weightSum = 0.f;
    }

    //Init with full precision
    __init(const uint4 packed){
        M = packed.x;
        targetFunc = asfloat(packed.y);
        weightSum = asfloat(packed.z);
        //Invalidate if nan or inf is present
        if(isinf(weightSum) || isnan(weightSum)){
            weightSum = 0;
            M = 0;
        }
    }

    //Init with reduced targedPDF precision
    __init(const uint2 packed){
        M = packed.x & 0xFFFF;
        targetFunc = f16tof32(packed.x >> 16);
        weightSum = asfloat(packed.y);
        //Invalidate if nan or inf is present
        if(isinf(weightSum) || isnan(weightSum)){
            weightSum = 0;
            M = 0;
        }
    }

    //Pack for uint texture
    uint4 pack(){        
        return uint4(M, asuint(targetFunc), asuint(weightSum), 0);
    }

    uint2 packReduced()
    {
        return uint2((M & 0xFFFF) | (f32tof16(targetFunc) << 16), asuint(weightSum));
    }
    

    //Updates reservoir with new light sample. Returns true if the light sample was selected
    [mutating] bool updateReservoir(const float invSelectPdf,const float lstargetFunc, const float rnd){
        float risWeight = lstargetFunc * invSelectPdf;
        weightSum += risWeight;
        M += 1;
        bool selectSample = (rnd * weightSum < risWeight);
    
        if(selectSample){
            targetFunc = lstargetFunc;
        }
        return selectSample;
    }

    //Combines another reservoir with this reservoir. Returs true if the new reservoir was selected
    [mutating] bool combineReservoir(in const Reservoir other, const float othertargetFunc,const float rnd){
        float normalizationFactor = other.weightSum * other.M;

        float risWeight = othertargetFunc * normalizationFactor;
    
        M += other.M;
    
        weightSum += risWeight;

        bool selectSample = (rnd * weightSum < risWeight);
    
        if(selectSample){
            targetFunc = othertargetFunc;
        }

        return selectSample;
    }

    // Stochastic Pairwise MIS: 合并 Reservoir 时内联计算 pairwise MIS 权重
    // p_hat_q: 样本在当前像素处的 target function 值
    // p_hat_neighbor: 样本在邻居像素处的 target function 值
    // M_q: 当前像素 Reservoir 的 M 值（合并前）
    [mutating] bool combineReservoirPairwise(in const Reservoir other, const float p_hat_q, const float p_hat_neighbor, const float M_q, const float rnd){
        // 计算 Pairwise MIS 权重: w = p_hat_q / (p_hat_q + (M_neighbor / M_q) * p_hat_neighbor)
        float m_ratio = (M_q > 0) ? (float(other.M) / max(1.0, M_q)) : 0.0;
        float denom = p_hat_q + m_ratio * p_hat_neighbor;
        float pairwiseWeight = (denom > 1e-8) ? (p_hat_q / denom) : 0.0;

        float risWeight = p_hat_q * other.weightSum * other.M * pairwiseWeight;

        M += other.M;
        weightSum += risWeight;

        bool selectSample = (rnd * weightSum < risWeight);

        if(selectSample){
            targetFunc = p_hat_q;
        }

        return selectSample;
    }

    //Finalizes a sample. Normalizes the Wsum to W (and uses weightSum to store W)
    [mutating] void finalizeSample(float normalizationNumerator, float normalizationDenominator){
        float denominator = targetFunc * normalizationDenominator;
        
        if(denominator > 0)
            weightSum = (weightSum * normalizationNumerator) / denominator;
        else
            weightSum = 0;
    }
};
