import Foundation
import MLX
import MLXNN
import MLXLMCommon

private struct Config: Codable {
    let formatVersion: String
    let architecture: Arch?

    struct Arch: Codable {
        let numExperts: Int?
        enum CodingKeys: String, CodingKey { case numExperts = "num_experts" }
    }

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case architecture
    }
}

func isJANG(at dir: URL) -> Bool {
    FileManager.default.fileExists(atPath: dir.appendingPathComponent("jang_config.json").path)
}

private func loadConfig(at dir: URL) -> Config? {
    guard let data = try? Data(contentsOf: dir.appendingPathComponent("jang_config.json")) else { return nil }
    return try? JSONDecoder().decode(Config.self, from: data)
}

/*
 Infers actual quantization bits from tensor shapes.
 JANG v2 stores mixed-precision weights in standard MLX safetensors format.
 The config.json declares a single nominal bits value, but each layer may use
 a different bit width. This function recovers the true bits by examining the
 packed uint32 weight dimensions vs the scales dimensions.

 weight: [..., packedCols] where packedCols = inFeatures * bits / 32
 scales: [..., nGroups]    where nGroups    = inFeatures / groupSize
 → bits = packedCols * 32 / (nGroups * groupSize)
*/
private func inferBits(weight: MLXArray, scales: MLXArray) -> (bits: Int, groupSize: Int)? {
    guard let wLast = weight.shape.last, let sLast = scales.shape.last, wLast > 0, sLast > 0 else { return nil }
    for bits in [2, 3, 4, 6, 8] {
        let elemsPerU32 = 32 / bits
        let inFeatures = wLast * elemsPerU32
        if inFeatures % sLast == 0 {
            return (bits, inFeatures / sLast)
        }
    }
    return nil
}

/*
 Drop-in replacement for QuantizedSwitchLinear that carries the correct per-layer
 bit width. SwitchLinear.weight / QuantizedSwitchLinear.scales / biases are internal
 to mlx-swift-lm, so all tensors are read via Module.parameters() and stored locally.
*/
final class JANGSwitchLinear: SwitchLinear {
    private let jangWeight: MLXArray
    private let jangScales: MLXArray
    private let jangBiases: MLXArray?
    private let jangBias: MLXArray?
    let groupSize: Int
    let bits: Int

    init(from source: QuantizedSwitchLinear, groupSize: Int, bits: Int) {
        let params = source.parameters().flattened()
        let w = params.first(where: { $0.0 == "weight" })!.1
        let b = params.first(where: { $0.0 == "bias" })?.1
        let s = params.first(where: { $0.0 == "scales" })!.1
        let bs = params.first(where: { $0.0 == "biases" })?.1

        jangWeight = w
        jangScales = s
        jangBiases = bs
        jangBias = b
        self.groupSize = groupSize
        self.bits = bits

        super.init(
            inputDims: source.inputDims,
            outputDims: source.outputDims,
            numExperts: source.numExperts,
            weight: w,
            bias: b
        )
        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x, jangWeight,
            scales: jangScales,
            biases: jangBiases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            sortedIndices: sortedIndices
        )
        if let bias = jangBias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }
        return result
    }
}

/*
 Walks all leaf modules and replaces any QuantizedLinear or QuantizedSwitchLinear
 whose declared bits do not match the actual packed tensor shapes. This is required
 for JANG v2 models: config.json carries only the minimum bits used, but each layer
 may be quantized at a different bit width (e.g. attention at 8-bit, experts at 2-bit).
*/
func patchJANG(model: Module, at dir: URL) {
    guard isJANG(at: dir) else { return }

    var updates: [(String, Module)] = []

    for (path, module) in model.leafModules().flattened() {
        if let ql = module as? QuantizedLinear {
            guard
                let (bits, gs) = inferBits(weight: ql.weight, scales: ql.scales),
                bits != ql.bits || gs != ql.groupSize
            else { continue }
            updates.append((path, QuantizedLinear(
                weight: ql.weight, bias: ql.bias,
                scales: ql.scales, biases: ql.biases,
                groupSize: gs, bits: bits
            )))
        } else if let qsl = module as? QuantizedSwitchLinear {
            let params = qsl.parameters().flattened()
            guard
                let weight = params.first(where: { $0.0 == "weight" })?.1,
                let scales = params.first(where: { $0.0 == "scales" })?.1,
                let (bits, gs) = inferBits(weight: weight, scales: scales),
                bits != qsl.bits || gs != qsl.groupSize
            else { continue }
            updates.append((path, JANGSwitchLinear(from: qsl, groupSize: gs, bits: bits)))
        }
    }

    if !updates.isEmpty {
        model.update(modules: ModuleChildren.unflattened(updates))
    }
}
