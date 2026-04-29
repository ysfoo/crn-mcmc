using Downloads
DATA_URL = "https://raw.githubusercontent.com/ProfMJSimpson/SigmoidGrowth/main/Data.xlsx";
DATA_FNAME = joinpath(@__DIR__, "data.xlsx");
Downloads.download(DATA_URL, DATA_FNAME);

