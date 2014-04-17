local brickpi = require "brickpi"
--
local images_dir = "%simages"%{ arg[0]:get_path() }
--
local filename            = arg[1] or "nets/default.net"
local perturbation_seed   = 5678
local learning_rate       = 0.1
local momentum            = 0.2
local weight_decay        = 1e-04
local L1_norm             = 1e-05
local max_norm_penalty    = 4
local perturbation_random = random(perturbation_seed)

-- BrickPi params
local BLACK            = 500
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
local thenet = trainer:get_component()
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
  trainer:save(filename, "binary")
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
