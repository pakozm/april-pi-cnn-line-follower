local brickpi = require "brickpi"
--
local images_dir = "%simages"%{ arg[0]:get_path() }
--
local filename            = arg[1] or "nets/default.net"
local out_filename        = arg[2] or "nets/last.net"
local perturbation_seed   = 5678
local learning_rate       = 0.1
local momentum            = 0.2
local weight_decay        = 1e-04
local L1_norm             = 1e-05
local max_norm_penalty    = 4
local perturbation_random = random(perturbation_seed)

-- BrickPi params
local SPEED            = 0.18
local BLACK            = 560
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
local PENALTY = -10
local REWARD  =   1
local EPSILON = 0.2
local SEED    = 85427
local exploration_random = random(SEED)

--
local trainer = util.deserialize(filename)
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
local thenet  = trainer:get_component()
local weights = trainer:get_weights_table()
--
local optimizer = trainer:get_optimizer()

-- FUNCTIONS

function sleep(v)
  os.execute("sleep %f"%{v})
end

function do_until(f)
  while not f() do sleep(0.01) end
end

function calibrate()
  print("CALIBRATING BLACK...")
  local mean_var = stats.mean_var()
  local N = 100
  for i=1,N do
    do_until(brickpi.setup)
    local value = brickpi.sensorValue(LIGHT_SENSOR)
    mean_var:add(value)
    sleep(0.1)
  end
  local mean,var = mean_var:compute()
  BLACK = mean - 2*var
  print("BLACK IS: ", BLACK)
end

function take_image()
  local command = "ls -t %s/*" % {images_dir}
  local g = assert(io.popen(command, "r"))
  -- remove last images, because they could be corrupted
  for i=1,3 do g:read("*l") end
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
  local reward
  if is_black(value) then
    reward = REWARD
  else
    reward = PENALTY
  end
  acc_value = acc_value * HISTORY_DISCOUNT + (1 - HISTORY_DISCOUNT) * reward
  return acc_value,value
end

function setup_brickpi()
  do_until(brickpi.setup)
  brickpi.motorEnable(LEFT_MOTOR, RIGHT_MOTOR)
  brickpi.sensorType(LIGHT_SENSOR, brickpi.TYPE_SENSOR_LIGHT_ON)
  brickpi.setupSensors()
  do_until(brickpi.update)
  sleep(1.0)
end

function do_action(action)
  if action == ACTION_FORWARD then
    brickpi.motorSteering(LEFT_MOTOR, RIGHT_MOTOR,  0, SPEED*1.5)
  elseif action == ACTION_LEFT then
    brickpi.motorSteering(LEFT_MOTOR, RIGHT_MOTOR, -1, SPEED)
  elseif action == ACTION_RIGHT then
    brickpi.motorSteering(LEFT_MOTOR, RIGHT_MOTOR,  1, SPEED)
  else
    error("Uknown action= " .. action)
  end
end

function take_action(output)
  local coin = exploration_random:rand()
  if coin < EPSILON then
    return exploration_random:choose{ACTION_FORWARD,ACTION_LEFT,ACTION_RIGHT}
  end
  local _,argmax = output:max()
  return argmax
end

local gradients = matrix.dict()
function update(prev_output, prev_action, state, reward)
  local loss,output
  loss,gradients,output =
    optimizer:execute(function(it)
                        thenet:reset(it)
                        local output = thenet:forward(state):get_matrix()
                        local error_grad = matrix.col_major(1, NACTIONS):zeros()
                        local qsa = prev_output:get(1, prev_action)
                        local expected_qsa = qsa + ALPHA * ( reward + DISCOUNT * output:max() - qsa )
                        local loss = (qsa - expected_qsa)^2
                        error_grad:set(1, prev_action, 0.5 * ( qsa - expected_qsa ) )
                        thenet:backprop(error_grad)
                        gradients:zeros()
                        gradients = thenet:compute_gradients(gradients)
                        return loss,gradients,output
                      end,
                      weights)
  return loss,output
end

-- MAIN

setup_brickpi()
calibrate()

local finished = false
signal.register(signal.SIGINT, function() finished = true end)

local prev_output = matrix.col_major(1,NACTIONS):zeros()
local prev_action = ACTION_FORWARD
local clock = util.stopwatch()
while not finished do
  clock:reset()
  clock:go()
  --
  local img_path  = take_image()
  local input_img = ImageIO.read(img_path)
  local input = normalize(input_img:matrix():clone("col_major"))
  input = input:rewrap(1, table.unpack(input:dim()))
  local reward,sensor_value = compute_reward()
  local loss,output = update(prev_output, prev_action, input, reward)
  trainer:save(out_filename, "binary")
  local action = take_action(output)
  do_action(action)
  do_until(brickpi.update)
  --
  prev_output = output
  prev_action = action
  --
  clock:stop()
  local t1,t2 = clock:read()
  printf("OUTPUT: %8.2f %8.2f %8.2f  ACTION: %d  SENSOR: %4d (%4d)  REWARD: %6.2f  LOSS: %8.4f  TIME: %5.2f %5.2f\n",
         output:get(1,1), output:get(1,2), output:get(1,3),
         action, sensor_value, BLACK, reward, loss, t1, t2)
  local extra_sleep = SLEEP - t1
  if extra_sleep > 0 then
    sleep(extra_sleep)
  end
end
