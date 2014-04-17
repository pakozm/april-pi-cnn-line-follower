local brickpi = require "brickpi"
--
local images_dir = "%simages"%{ arg[0]:get_path() }
--
local weights_seed        = 1234
local perturbation_seed   = 5678
local w_inf               = -0.1
local w_sup               =  0.1
local learning_rate       = 0.1
local momentum            = 0.2
local weight_decay        = 1e-04
local L1_norm             = 1e-05
local max_norm_penalty    = 4
local ann_type            = "perceptron" -- perceptron, mlp, cnn
local activation_function = "relu"
--
local weights_random      = random(weights_seed)
local perturbation_random = random(perturbation_seed)

-- BrickPi params
local BLACK            = 0.5
local HISTORY_DISCOUNT = 0.5
local SLEEP            = 0.1
local LEFT_MOTOR       = brickpi.PORT_A
local RIGHT_MOTOR      = brickpi.PORT_D
local LIGHT_SENSOR     = brickpi.PORT_1

-- QLearning parameters
local ACTION_FORWARD  = 1
local ACTION_LEFT     = 2
local ACTION_RIGHT    = 3
local NACTIONS        = 3
local DISCOUNT        = 0.9
local ALPHA           = 0.1
local PENALTY = -1
local REWARD  =  1

-- ANN topology
local PLANES=3
local HEIGHT=32
local WIDTH=32
local INPUT_SIZES = { HEIGHT, WIDTH, PLANES }
local SIZE = iterator(ipairs(INPUT_SIZES)):select(2):reduce(math.mul(),1)
-- RGB image ( 3 input planes, 5x5 kernel ) => nconv1 output planes
local conv1  = {5, 5, 3}
local nconv1 = 10
-- max-pooling 2x2 kernel
local maxp1  = {1, 2, 2}
-- ( nconv1 input planes, 3x3 kernel ) => nconv2 output planes
local conv2  = {nconv1, 3, 3}
local nconv2 = 20
-- max-pooling 2x2 kernel
local maxp2  = {1, 2, 2}
local hidden = 100

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
trainer:set_option("learning_rate",     learning_rate)
trainer:set_option("momentum",          momentum)
trainer:set_option("weight_decay",      weight_decay)
trainer:set_option("L1_norm",           L1_norm)
trainer:set_option("max_norm_penalty",  max_norm_penalty)
--
trainer:set_layerwise_option("b.", "weight_decay",     0.0)
trainer:set_layerwise_option("b.", "max_norm_penalty", 0.0)
trainer:set_layerwise_option("b.", "L1_norm",          0.0)
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
local optimizer = trainer:get_optimizer()

-- FUNCTIONS

function take_image()
  local command = "ls -t %s/*" % {images_dir}
  local g = assert(io.popen(command, "r"))
  -- remove two last images, because they could be corrupted
  for i=1,2 do g:read("*l") end
  -- take the third image
  local img_path = g:read("*l")
  g:close()
  return img_path
end

function normalize(m)
  local sz   = m:dim(1)*m:dim(2)
  local mp   = m:rewrap(sz, m:dim(3))
  local sums = mp:sum(1):scal(1/sz):toTable()
  mp(':',1):scalar_add(-sums[1])
  mp(':',2):scalar_add(-sums[2])
  mp(':',3):scalar_add(-sums[3])
  return m
end

function is_black(v)
  if v > BLACK then return true end
end

local acc_value = 0
function compute_reward()
  local value  = brickpi.sensorValue(LIGHT_SENSOR)
  print(value)
  local reward
  if is_black(value) then
    reward = REWARD
  else
    reward = PENALTY
  end
  acc_value = acc_value * HISTORY_DISCOUNT + (1 - HISTORY_DISCOUNT) * reward
  return acc_value
end

function setup_brickpi()
  assert(brickpi.setup())
  brickpi.motorEnable(LEFT_MOTOR, RIGHT_MOTOR)
  brickpi.sensorType(LIGHT_SENSOR, brickpi.TYPE_SENSOR_LIGHT_ON)
  brickpi.setupSensors()
end

function do_action(action)
  if action == ACTION_FORWARD then
    brickpi.motorSteering(LEFT_MOTOR, RIGHT_MOTOR,  0, 0.7)
  elseif action == ACTION_LEFT then
    brickpi.motorSteering(LEFT_MOTOR, RIGHT_MOTOR, -1, 0.7)
  elseif action == ACTION_RIGHT then
    brickpi.motorSteering(LEFT_MOTOR, RIGHT_MOTOR,  1, 0.7)
  else
    error("Uknown action= " .. action)
  end
end

function update()
end

-- MAIN

setup_brickpi()

local prev_output = matrix.col_major(1,NACTIONS):zeros()
local prev_action = FORWARD
local clock = util.stopwatch()
while true do
  clock:reset()
  clock:go()
  --
  local img_path  = take_image()
  local input_img = ImageIO.read(img_path)
  local input = normalize(input_img:matrix():clone("col_major"))
  input = input:rewrap(1, table.unpack(input:dim()))
  local output = thenet:forward(input):get_matrix()
  local reward = compute_reward()
  update(prev_output, prev_action, output, reward)
  --
  prev_output = output
  prev_action = action
  --
  clock:stop()
  local t1,t2 = clock:read()
  printf("TIME: %.2f %.2f\n", t1, t2)
  local sleep = SLEEP - t1
  if sleep > 0 then
    printf("SLEEP: %.2f\n", sleep)
    brickpi.sleep(sleep)
  end
  assert(brickpi.update())
end
