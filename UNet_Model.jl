using Flux

function concat(mx, x)
    return cat(x, mx, dims = 3)
end

Create_Modified_Unet_Model() = Chain(
    BatchNorm(1),
    Conv((3, 3), 1 => 16, pad = 1, relu),
    BatchNorm(16),
    Conv((3, 3), 16 => 16, pad = 1, relu),
    SkipConnection(
        Chain(
            MaxPool((2, 2)),
            BatchNorm(16),
            Conv((3, 3), 16 => 32, pad = 1, relu),
            BatchNorm(32),
            Conv((3, 3), 32 => 32, pad = 1, relu),
            SkipConnection(
                Chain(
                    MaxPool((2, 2)),
                    BatchNorm(32),
                    Conv((3, 3), 32 => 64, pad = 1, relu),
                    BatchNorm(64),
                    Conv((3, 3), 64 => 64, pad = 1, relu),
                    SkipConnection(
                        Chain(
                            MaxPool((2, 2)),
                            BatchNorm(64),
                            Conv((3, 3), 64 => 128, pad = 1, relu),
                            BatchNorm(128),
                            Conv((3, 3), 128 => 128, pad = 1, relu),
                            SkipConnection(
                                Chain(
                                    MaxPool((2, 2)),
                                    BatchNorm(128),
                                    Conv((3, 3), 128 => 256, pad = 1, relu),
                                    BatchNorm(256),
                                    Conv((3, 3), 256 => 256, pad = 1, relu),
                                    ConvTranspose((2, 2), 256 => 128, stride = 2)
                                ),
                                concat
                            ),
                            BatchNorm(256),
                            Conv((3, 3), 256 => 128, pad = 1, relu),
                            BatchNorm(128),
                            Conv((3, 3), 128 => 128, pad = 1, relu),
                            ConvTranspose((2, 2), 128 => 64, stride = 2)
                        ),
                        concat
                    ),
                    BatchNorm(128),
                    Conv((3, 3), 128 => 64, pad = 1, relu),
                    BatchNorm(64),
                    Conv((3, 3), 64 => 64, pad = 1, relu),
                    ConvTranspose((2, 2), 64 => 32, stride = 2),
                ),
                concat
            ),
            BatchNorm(64),
            Conv((3, 3), 64 => 32, pad = 1, relu),
            BatchNorm(32),
            Conv((3, 3), 32 => 32, pad = 1, relu),
            ConvTranspose((2, 2), 32 => 16, stride = 2),
        ),
        concat
    ),
    BatchNorm(32),
    Conv((3, 3), 32 => 16, pad = 1, relu),
    BatchNorm(16),
    Conv((3, 3), 16 => 16, pad = 1, relu),
    Conv((1, 1), 16 => 1, σ)
)
