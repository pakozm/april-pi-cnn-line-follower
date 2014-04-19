local brickpi = require "brickpi"
local utils = {
  ACTION_FORWARD  = 1,
  ACTION_LEFT     = 2,
  ACTION_RIGHT    = 3,
  NACTIONS        = 3,
  --
  ACTION_STOP     = 4,
  --
  EPSILON = 0.2,
  --
  LEFT_MOTOR   = brickpi.PORT_A,
  RIGHT_MOTOR  = brickpi.PORT_D,
  LIGHT_SENSOR = brickpi.PORT_1,
  --
  SPEED            = 100,
  SLEEP            = 0.1,
}

local perturbation_seed   = 5678
local learning_rate       = 0.01
local momentum            = 0.1
local weight_decay        = 1e-04
local L1_norm             = 0.0 -- 1e-05
local max_norm_penalty    = 4
local perturbation_random = random(perturbation_seed)

-------------------------------------------------------

function utils.take_image(images_dir)
  local command = "ls -t %s/*" % {images_dir}
  local g = assert(io.popen(command, "r"))
  -- remove last images, because they could be corrupted
  for i=1,3 do g:read("*l") end
  -- take the third image
  local img_path = g:read("*l")
  g:close()
  return img_path
end

function utils.setup_brickpi()
  utils.do_until(brickpi.setup)
  brickpi.motorEnable(utils.LEFT_MOTOR, utils.RIGHT_MOTOR)
  brickpi.sensorType(utils.LIGHT_SENSOR, brickpi.TYPE_SENSOR_LIGHT_ON)
  brickpi.setupSensors()
  utils.do_until(brickpi.update)
  utils.sleep(1.0)
end

function utils.do_action(action)
  if action == utils.ACTION_FORWARD then
    brickpi.motorSpeed(utils.LEFT_MOTOR, utils.RIGHT_MOTOR, utils.SPEED)
  elseif action == utils.ACTION_LEFT then
    brickpi.motorSpeed(utils.LEFT_MOTOR,  -utils.SPEED*0.7)
    brickpi.motorSpeed(utils.RIGHT_MOTOR,  utils.SPEED*0.7)
  elseif action == utils.ACTION_RIGHT then
    brickpi.motorSpeed(utils.LEFT_MOTOR,   utils.SPEED*0.7)
    brickpi.motorSpeed(utils.RIGHT_MOTOR, -utils.SPEED*0.7)
  elseif action == utils.ACTION_STOP then
    brickpi.motorSpeed(utils.LEFT_MOTOR, utils.RIGHT_MOTOR, 0)
  else
    error("Uknown action= " .. action)
  end
end

function utils.normalize(m)
  local d = m:dim()
  if #d == 3 then
    local sz   = d[1]*d[2]
    local mp   = m:rewrap(sz, d[3])
    local sums = mp:sum(1):scal(1/sz):toTable()
    mp(':',1):scalar_add(-sums[1])
    mp(':',2):scalar_add(-sums[2])
    mp(':',3):scalar_add(-sums[3])
  else
    m:scalar_add(m:sum()/m:size())
  end
  return m
end

function utils.get_input_from_image_path(img_path)
  local input_img = ImageIO.read(img_path)
  local input = utils.normalize(input_img:to_grayscale():invert_colors():matrix():clone("col_major"))
  local d = input:dim()
  input = input:rewrap(1, d[1], d[2], ( (#d == 3) and d[3] ) or 1)
  return input,input_img
end

function utils.do_until(f)
  while not f() do utils.sleep(0.01) end
end

function utils.sleep(v)
  os.execute("sleep %f"%{v})
end

-------------------------------------------------------

local sensor = {}

local function is_black(v,mean,var)
  if v > (mean-var) and v < (mean+var) then return true end
  return false
end

function sensor:compute_reward()
  local value  = brickpi.sensorValue(self.which_sensor)
  local reward
  if is_black(value,self.BLACK_MEAN,self.BLACK_V) then
    local x0 = self.BLACK_MEAN - self.BLACK_V
    local y = math.abs(value - x0) * self.slope + self.PENALTY
    reward = math.min(self.REWARD, y)
  else
    reward = self.PENALTY
  end
  self.value = value
  return reward,value
end

function sensor:calibrate()
  print("CALIBRATING BLACK...")
  local mean_var = stats.mean_var()
  local N = 100
  for i=1,N do
    utils.do_until(brickpi.update)
    local value = brickpi.sensorValue(LIGHT_SENSOR)
    mean_var:add(value)
    utils.sleep(0.1)
  end
  local mean,var = mean_var:compute()
  self.BLACK_MEAN = mean
  self.BLACK_V = math.max(20, math.min(30, 3*var))
  print("BLACK IS: ", self.BLACK_MEAN - self.BLACK_V,
        self.BLACK_MEAN + self.BLACK_V)
  self.slope = (self.REWARD - self.PENALTY) / (self.BLACK_V * 2)
end

function sensor:__call(which_sensor,REWARD,PENALTY)
  local obj = {
    which_sensor = which_sensor,
    REWARD = REWARD,
    PENALTY = PENALTY,
  }
  setmetatable(obj, { __index=self })
  return obj
end
setmetatable(sensor,sensor)

-------------------------------------------------------

local exploration_random = random()
local function take_action(output)
  local coin = exploration_random:rand()
  if coin < utils.EPSILON then
    return exploration_random:choose{utils.ACTION_FORWARD,
                                     utils.ACTION_LEFT,
                                     utils.ACTION_RIGHT}
  end
  local _,argmax = output:max()
  return argmax
end

local trainer = {}

local perturbation_random = random()
local noise = ann.components.salt_and_pepper{ prob=0.2, zero=0, one=1,
                                              random=perturbation_random }
function trainer:update(prev_state, prev_action, state, reward)
  local prev_state = noise:forward(prev_state):get_matrix()
  local state = noise:forward(state):get_matrix()
  local thenet = self.thenet
  local optimizer = self.optimizer
  local gradients = self.gradients
  local traces = self.traces
  local error_grad = matrix.col_major(1, utils.NACTIONS):zeros()
  local loss,output,qs
  loss,gradients,output,qs,expected_qsa =
    optimizer:execute(function(it)
                        assert(not it or it == 0)
                        thenet:reset(it)
                        local output = thenet:forward(state):get_matrix()
                        local qs  = thenet:forward(prev_state,true):get_matrix()
                        local qsa = qs:get(1, prev_action)
                        local expected_qsa = math.min(1, math.max(0, reward + self.DISCOUNT * output:max()))
                        local diff = (qsa - expected_qsa)
                        local loss = 0.5 * diff * diff
                        error_grad:set(1, prev_action, ( qsa - expected_qsa ) )
                        thenet:backprop(error_grad)
                        gradients:zeros()
                        gradients = thenet:compute_gradients(gradients)
                        if traces:size() == 0 then
                          for name,g in pairs(gradients) do
                            traces[name] = matrix.as(g):zeros()
                          end
                        end
                        traces:scal(0.5)
                        traces:axpy(1.0, gradients)
                        return loss,traces,output,qs,expected_qsa
                      end,
                      weights)
  self.gradients = gradients
  return loss,output,qs,expected_qsa
end

function trainer:save(out_filename)
  self.tr:save(out_filename, "binary")
end

function trainer:one_step(img_path, sensor)
  local input,input_img = utils.get_input_from_image_path(img_path)
  self.input_img = input_img
  local reward,sensor_value = sensor:compute_reward()
  local loss,output,expected_qsa
  local action
  if self.prev_input and self.prev_action then
    loss,output,qs,expected_qsa = self:update(self.prev_input, self.prev_action,
                                              input, reward)
    action = take_action(output)
    printf("Q(s): %8.2f %8.2f %8.2f  E(Q(s)): %8.2f   ACTION: %d  SENSOR: %4d (%4d %4d) REWARD: %6.2f  LOSS: %8.4f  MP: %.4f %.4f\n",
           qs:get(1,1), qs:get(1,2), qs:get(1,3), expected_qsa,
           self.prev_action, sensor_value, sensor.BLACK_MEAN - sensor.BLACK_V,
           sensor.BLACK_MEAN + sensor.BLACK_V, reward, loss,
           self.tr:norm2("w."), self.tr:norm2("b."))
  else
    action = utils.ACTION_FORWARD
  end
  self.prev_input  = input
  self.prev_action = action
  return action
end

function trainer:__call(filename, DISCOUNT)
  local tr = util.deserialize(filename)
  tr:set_option("learning_rate",     learning_rate)
  tr:set_option("momentum",          momentum)
  tr:set_option("weight_decay",      weight_decay)
  tr:set_option("L1_norm",           L1_norm)
  tr:set_option("max_norm_penalty",  max_norm_penalty)
  --
  tr:set_layerwise_option("b.", "weight_decay",     0.0)
  tr:set_layerwise_option("b.", "max_norm_penalty", 0.0)
  tr:set_layerwise_option("b.", "L1_norm",          0.0)
  --
  local thenet  = tr:get_component()
  local weights = tr:get_weights_table()
  --
  local optimizer = tr:get_optimizer()
  local obj = {
    tr = tr,
    thenet = thenet,
    weights = weights,
    optimizer = optimizer,
    gradients = matrix.dict(),
    traces = matrix.dict(),
    DISCOUNT = DISCOUNT,
  }
  setmetatable(obj, { __index=self })
  return obj
end
setmetatable(trainer,trainer)

-------------------------------------------------------

local offline_controller = {}

function offline_controller:next()
  self.IDX=self.IDX+1
  local info_f = io.open("%s/info%06d.txt"%{self.dir,self.IDX},"r")
  if not info_f then return false end
  local info_t = info_f:read("*l"):tokenize()
  info_f:close()
  self.info = {
    action = tonumber(info_t[1]),
    sensor = tonumber(info_t[2]),
    mean = tonumber(info_t[4]),
    var = tonumber(info_t[6]),
  }
  self.input = ImageIO.read("%s/info%06d.png"%{self.dir,self.IDX})
  if not self.input then return false end
  return true
end

function offline_controller:get_input() return self.input end
function offline_controller:get_info() return self.info end

function offline_controller:__call(dir)
  local obj = { dir=dir, IDX=0 }
  setmetatable(obj, { __index=self })
  return obj
end
setmetatable(offline_controller,offline_controller)

-------------------------------------------------------

utils.trainer = trainer
utils.sensor  = sensor
utils.offline_controller = offline_controller

return utils
