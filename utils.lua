local utils = {}

function utils.normalize(m)
  local sz   = m:dim(1)*m:dim(2)
  local mp   = m:rewrap(sz, m:dim(3))
  local sums = mp:sum(1):scal(1/sz):toTable()
  mp(':',1):scalar_add(-sums[1])
  mp(':',2):scalar_add(-sums[2])
  mp(':',3):scalar_add(-sums[3])
  return m
end

return utils
