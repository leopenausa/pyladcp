-- Pandoc Lua filter: render MkDocs-Material admonitions in the PDF.
--
-- In the site, `!!! tip "Title"` + an indented body is a styled box. Plain
-- pandoc sees the marker as a paragraph and the indented body as a code
-- block. This filter rewrites the pair into a blockquote with a bold title.

local PREFIX = "!!!"

local function admonition_title(para)
  -- para.content like: Str("!!!") Space Str("tip") Space Quoted(...) ...
  local first = para.content[1]
  if not first or first.t ~= "Str" or first.text ~= PREFIX then
    return nil
  end
  local kind, title = nil, nil
  for i = 2, #para.content do
    local el = para.content[i]
    if el.t == "Str" and not kind then
      kind = el.text
    elseif el.t == "Quoted" then
      title = pandoc.utils.stringify(el.content)
    end
  end
  if not kind then return nil end
  title = title or (kind:sub(1, 1):upper() .. kind:sub(2))
  return title
end

function Blocks(blocks)
  local out = pandoc.Blocks({})
  local i = 1
  while i <= #blocks do
    local b = blocks[i]
    local title = (b.t == "Para") and admonition_title(b) or nil
    if title then
      local body = pandoc.Blocks({ pandoc.Para({ pandoc.Strong({ pandoc.Str(title) }) }) })
      local nxt = blocks[i + 1]
      if nxt and nxt.t == "CodeBlock" then
        -- the indented admonition body: re-read it as markdown
        body:extend(pandoc.read(nxt.text, "markdown").blocks)
        i = i + 1
      end
      out:insert(pandoc.BlockQuote(body))
    else
      out:insert(b)
    end
    i = i + 1
  end
  return out
end
