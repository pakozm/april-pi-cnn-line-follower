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
local learning_rate       = 0.1
local momentum            = 0.4
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

local strategies = trainable.qlearning_trainer.strategies
local exploration_random = random()
local take_action = strategies.make_epsilon_greedy(utils.EPSILON,
                                                   exploration_random)

local trainer = {}

function trainer:save(out_filename)
  self.tr:save(out_filename, "binary")
end

local perturbation_random = random()
local noise = ann.components.salt_and_pepper{ prob=0.2, zero=0, one=1,
                                              random=perturbation_random }
function trainer:train_batch()
  local batch = assert(self.batch)
  -- BATCH TRAIN
  local in_ds,out_ds,mask_ds = batch:compute_dataset_pair()
  -- for ipat,pat in out_ds:patterns() do print(ipat, table.concat(pat, " "), table.concat(mask_ds:getPattern(ipat), " ")) end
  local train_func = trainable.train_wo_validation{
    max_epochs = 100,
    min_epochs = 20,
    percentage_stopping_criterion = 0.01,
  }
  local in_ds = dataset.token.filter(in_ds, noise)
  while train_func:execute(function()
                             local tr_error = sup_trainer:train_dataset{
                               input_dataset  = in_ds,
                               output_dataset = out_ds,
                               mask_dataset = mask_ds,
                               shuffle = shuffle_random,
                             }
                             return sup_trainer,tr_error
                           end) do
    print(train_func:get_state_string())
  end
  self.batch = self.ql:get_batch_builder()
end

function trainer:append_one_step(img_path, sensor)
  local input,input_img = utils.get_input_from_image_path(img_path)
  self.input_img = input_img
  local reward,sensor_value = sensor:compute_reward()
  local output = self.ql:calculate(self.input)
  local action
  if self.prev_input and self.prev_action then
    local batch = self.batch or self.ql:get_batch_builder()
    batch:add(self.prev_input, self.prev_output, self.prev_action, reward)
  else
    action = utils.ACTION_FORWARD
  end
  local action = take_action(output)
  self.prev_input  = input
  self.prev_action = action
  self.prev_output = output
  return action,reward
end

function trainer:reset()
  self.ql:reset()
  self.prev_action = nil
  self.prev_input  = nil
  self.prev_output = nil
end

function trainer:__call(filename, DISCOUNT)
  local tr = util.deserialize(filename)
  local thenet  = tr:get_component()
  local weights = tr:get_weights_table()
  local ql = trainable.qlearning_trainer{
    sup_trainer = tr,
    discount = DISCOUNT,
    clampQ = function(v) return math.clamp(v,0.0,1.0) end,
  }
  ql:set_option("learning_rate",     learning_rate)
  ql:set_option("momentum",          momentum)
  ql:set_option("weight_decay",      weight_decay)
  ql:set_option("L1_norm",           L1_norm)
  ql:set_option("max_norm_penalty",  max_norm_penalty)
  --
  ql:set_layerwise_option("b.", "weight_decay",     0.0)
  ql:set_layerwise_option("b.", "max_norm_penalty", 0.0)
  ql:set_layerwise_option("b.", "L1_norm",          0.0)
  --
  local obj = {
    tr = tr,
    ql = ql,
    thenet = thenet,
    weights = weights,
  }
  setmetatable(obj, { __index=self })
  return obj
end
setmetatable(trainer,trainer)

-------------------------------------------------------

local offline_controller = {}

function offline_controller:next()
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
  self.input_path = "%s/input%06d.png"%{self.dir,self.IDX+1}
  local aux = io.open(self.input_path,"r")
  if not aux then return false end
  aux:close()
  self.IDX=self.IDX+1
  return true
end

function offline_controller:get_input_path() return self.input_path end
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
