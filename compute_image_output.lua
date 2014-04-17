package.path = "%s?.lua;%s"%{ arg[0]:get_path(), package.path }
local utils = require "utils"
local image_filename = assert(arg[1], "Needs an image filename")
local filename = arg[2] or "nets/default.net"
local trainer = util.deserialize(filename)
local thenet = trainer:get_component()
local input_img = ImageIO.read(image_filename)
local input = utils.normalize(input_img:matrix():clone("col_major"))
input = input:rewrap(1, table.unpack(input:dim()))
output = thenet:forward(input):get_matrix()
print(output)
