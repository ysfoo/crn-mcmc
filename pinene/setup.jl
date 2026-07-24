# Fetch packages.
using Catalyst, Combinatorics
using DataFrames, DelimitedFiles, Distributions
using Logging, LoggingExtras

nowarn_logger = EarlyFilteredLogger(global_logger()) do log
    log.level != Logging.Warn
end

macro nowarn_load(filename, vars...)
    quote
        ($([esc(v) for v in vars]...),) =
            with_logger(nowarn_logger) do
                ($([:(load($(esc(filename)), $(string(v)))) for v in vars]...),)
            end

        $(Symbol[v for v in vars])
    end
end

# Generates all potential models.
begin
    t = default_t()
    
    # use labelling from Box et al. (1973)
    # X1 = α-pinene, X2 = dipentene, X3 = allo-ocimene, X4 = pyronene, X5 = dimer
    isomers = @species X1(t) X2(t) X3(t) X4(t);
    dimer = (@species X5(t))[1];
    species_vec = [isomers; dimer]

    rxs_no_k = [
        [([reactant], [product]) for product in isomers, reactant in isomers if reactant !== product];
        [([dimer], [isomer], [1], [2]) for isomer in isomers];
        [([isomer], [dimer], [2], [1]) for isomer in isomers];
    ];
    n_rx = length(rxs_no_k) # number of reactions
    @parameters k[1:n_rx] # reaction rate constants
    rx_vec = [
        Reaction(kval, rx_no_k...) for (rx_no_k, kval) in zip(rxs_no_k, k)
    ];

    # CRN
    @named model = ReactionSystem(rx_vec, t, species_vec, [k])
    model = complete(model)
end

# Read data
data_fname = joinpath(@__DIR__, "data.txt");
fullmat = readdlm(data_fname);
t_obs = fullmat[:,1];
data = fullmat[:,2:end];
data[:,5] ./= 2.0; 
n_obs, n_species = size(data);

t_span = extrema(t_obs);
x0_map = vcat([:X1 => 100.], [s.val.f.name => 0. for s in species_vec[2:end]])
base_oprob = ODEProblem(model, x0_map, t_span, [only(parameters(model)) => collect(1.0:20.0)]);

# Latex labels
rx_labels = [
        begin 
        s = string(rx)
        s = s[findfirst(' ', s)+1:end]
        s = Base.replace(s, "-->" => "\\rightarrow")
        s = Base.replace(s, "X" => "X_")
        s = Base.replace(s, "*" => "")
        s
    end for rx in rx_vec
];