module ChainRulesCoreBFloat16sExt

using ChainRulesCore
using BFloat16s: BFloat16

ChainRulesCore.ProjectTo(::BFloat16) = ProjectTo{BFloat16}()

end
