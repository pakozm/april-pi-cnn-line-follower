local filename     = arg[1] or "nets/default.net"
local ann_type     = arg[2] or "perceptron" -- perceptron, mlp, cnn
local weights_seed = 1234
--
local activation_function = "relu"
local w_inf               = -0.1
local w_sup               =  0.1
local weights_random      = random(weights_seed)

-- ANN topology
local PLANES=3
local HEIGHT=16
local WIDTH=16
local INPUT_SIZES = { HEIGHT, WIDTH, PLANES }
local SIZE = iterator(ipairs(INPUT_SIZES)):select(2):reduce(math.mul(),1)
local NACTIONS = 3

-- RGB image ( 3 input planes, 5x5 kernel ) => nconv1 output planes
local conv1  = {5, 5, 3}
local nconv1 = 8
-- max-pooling 2x2 kernel
local maxp1  = {1, 2, 2}
-- ( nconv1 input planes, 3x3 kernel ) => nconv2 output planes
local conv2  = {nconv1, 3, 3}
local nconv2 = 16
-- max-pooling 2x2 kernel
local maxp2  = {1, 2, 2}
local hidden = 32

local thenet = ann.components.stack()

if ann_type == "cnn" then
  thenet:push( ann.components.convolution{ kernel=conv1, n=nconv1,
                                           weights="w1",
                                           input_planes_dim = 3 } ):
  push( ann.components.convolution_bias{ n=nconv1, ndims=#conv1,
                                         weights="b1" } ):
  push( ann.components.actf[activation_function]() ):
  push( ann.components.max_pooling{ kernel=maxp1 } ):
  push( ann.components.convolution{ kernel=conv2, n=nconv2,
                                    weights="w2" } ):
  push( ann.components.convolution_bias{ n=nconv2, ndims=#conv2,
                                         weights="b2" } ):
  push( ann.components.actf[activation_function]() ):
  push( ann.components.max_pooling{ kernel=maxp2 } ):
  push( ann.components.flatten() )
  
  local convolution_output_size = thenet:precompute_output_size(INPUT_SIZES)[1]
  
  thenet:
  push( ann.components.hyperplane{ input=convolution_output_size,
                                   output=hidden,
                                   bias_weights="b3",
                                   dot_product_weights="w3" } ):
  push( ann.components.actf[activation_function]() ):
  push( ann.components.hyperplane{ input=hidden,
                                   output=NACTIONS,
                                   bias_weights="b4",
                                   dot_product_weights="w4" } )
elseif ann_type == "mlp" then
  thenet:push( ann.components.flatten() ):
  push( ann.components.hyperplane{ input=SIZE,
                                   output=hidden,
                                   bias_weights="b1",
                                   dot_product_weights="w1" } ):
  push( ann.components.actf[activation_function]() ):
  push( ann.components.hyperplane{ input=hidden,
                                   output=NACTIONS,
                                   bias_weights="b2",
                                   dot_product_weights="w2" } )
elseif ann_type == "perceptron" then
  thenet:push( ann.components.flatten() ):
  push( ann.components.hyperplane{ input=SIZE,
                                   output=NACTIONS,
                                   bias_weights="b1",
                                   dot_product_weights="w1" } )
else
  error("Unknown ann_type= " .. ann_type)
end

local trainer = trainable.supervised_trainer(thenet)
trainer:build()
--
if ann_type ~= "perceptron" then
  trainer:randomize_weights{
    name_match  = "w.*",
    random      = weights_random,
    inf         = w_inf,
    sup         = w_sup,
    use_fanin   = true,
    use_fanout  = true,
  }
  trainer:randomize_weights{
    name_match  = "b.*",
    random      = weights_random,
    inf         = 0,
    sup         = 0.2,
    use_fanin   = true,
    use_fanout  = true,
  }
else
  trainer:randomize_weights{
    random      = weights_random,
    inf         = w_inf,
    sup         = w_sup,
    use_fanin   = true,
    use_fanout  = true,
  }
end
--
if io.open(filename) then error("Unable to overwrite filename: " .. filename) end
trainer:save(filename, "binary")
