-- Pandoc Lua filter for the PathSnap report (LaTeX output).
-- Inline code  -> \code{...}: coloured monospace that may break after / _ . , -
-- Code blocks  -> listings environment (line wrapping, light background)
local map = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["_"] = "\\_\\allowbreak{}",
  ["%"] = "\\%", ["#"] = "\\#", ["&"] = "\\&", ["$"] = "\\$",
  ["~"] = "\\textasciitilde{}", ["^"] = "\\textasciicircum{}",
  ["/"] = "/\\allowbreak{}", ["."] = ".\\allowbreak{}", [","] = ",\\allowbreak{}", ["-"] = "-\\allowbreak{}",
}

local function esc(s)
  local out = {}
  for _, c in utf8.codes(s) do
    local ch = utf8.char(c)
    out[#out + 1] = map[ch] or ch
  end
  return table.concat(out)
end

function Code(el)
  return pandoc.RawInline("latex", "\\code{" .. esc(el.text) .. "}")
end

function CodeBlock(el)
  return pandoc.RawBlock("latex", "\\begin{lstlisting}\n" .. el.text .. "\n\\end{lstlisting}")
end
