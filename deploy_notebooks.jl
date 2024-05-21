using Literate

dirs = Dict("./"=>"./", "scripts"=>"notebooks", "scripts_solutions"=>"notebooks_solutions")

function deploy!(src, dst)
    current_file = splitpath(@__FILE__)[end]
    jl_files = filter(fl -> splitext(fl)[end] == ".jl" && splitpath(fl)[end] != current_file, readdir(src; join=true))

    for fl in jl_files
        println("File: $fl")
        Literate.notebook(fl, dst, credit=false, execute=false, mdstrings=true)
    end
    return
end

for (src, dst) in dirs
    deploy!(src, dst)
end
