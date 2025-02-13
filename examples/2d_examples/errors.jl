H = [1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5]
plt1 = Plots.plot()
plt2 = Plots.plot()

# l = 1, p = 0
l2h1error = [0.10695429648961015 0.22991324795748327;
0.016910862350160042 0.06831803984721779; 
0.010847479521195712 0.08033229808324488;
0.004644546604636679 0.0592245589315573;
0.002194946926490657 0.0425033625581446];

Plots.plot!(plt1, H, l2h1error[:,1], xaxis=:log10, yaxis=:log10, label="L² Error (l=1, p=0)", lw=3, lc=:blue);
Plots.scatter!(plt1, H, l2h1error[:,1], label="")
Plots.plot!(plt2, H, l2h1error[:,2], xaxis=:log10, yaxis=:log10, label="H¹ Error (l=1, p=0)", lw=3, lc=:blue);
Plots.scatter!(plt2, H, l2h1error[:,2], label="")

# l = 2, p = 0
l2h1error = [0.10695429648961015 0.22991324795748327;
0.009017856356127767 0.04792270106673038;
0.002385560260441666 0.027767393733159125;
0.0011317412366356567 0.02405262940020477;
0.0004907423980950219 0.017893945587174838]

Plots.plot!(plt1, H, l2h1error[:,1], xaxis=:log10, yaxis=:log10, label="L² Error (l=2, p=0)", lw=3, lc=:red);
Plots.scatter!(plt1, H, l2h1error[:,1], label="")
Plots.plot!(plt2, H, l2h1error[:,2], xaxis=:log10, yaxis=:log10, label="H¹ Error (l=2, p=0)", lw=3, lc=:red);
Plots.scatter!(plt2, H, l2h1error[:,2], label="")      

# l = 3, p = 0
l2h1error = [0.10695429648961015 0.22991324795748327;
0.009296041759002192 0.049065225735199954;
0.0012234811679658212 0.01367311469556407;
0.0003541923752540223 0.0085684475754577;
0.00013537089817940745 0.00640456568682608]

Plots.plot!(plt1, H, l2h1error[:,1], xaxis=:log10, yaxis=:log10, label="L² Error (l=3, p=0)", lw=3, lc=:olive);
Plots.scatter!(plt1, H, l2h1error[:,1], label="")
Plots.plot!(plt2, H, l2h1error[:,2], xaxis=:log10, yaxis=:log10, label="H¹ Error (l=3, p=0)", lw=3, lc=:olive);
Plots.scatter!(plt2, H, l2h1error[:,2], label="") 

# l = 4, p = 0

l2h1error = [0.10695429648961015 0.22991324795748327;
0.009296041759002192 0.049065225735199954;
0.0009947602694513084 0.01142733861936724;
0.00013910524396659206 0.0034835186517853892;
4.2625600467092376e-5 0.0021808139048485734]

Plots.plot!(plt1, H, l2h1error[:,1], xaxis=:log10, yaxis=:log10, label="L² Error (l=4, p=0)", lw=3, lc=:orange);
Plots.scatter!(plt1, H, l2h1error[:,1], label="")
Plots.plot!(plt2, H, l2h1error[:,2], xaxis=:log10, yaxis=:log10, label="H¹ Error (l=4, p=0)", lw=3, lc=:orange);
Plots.scatter!(plt2, H, l2h1error[:,2], label="")

##############

# l = 1, p = 0
l2h1error = [0.10695429648960116 0.22991324795746892;
0.2126926940462622 0.45027262315088823; 
0.7123715192062066 0.8435244025050337;
0.9272424265057524 0.9629086745822627;
0.9836592803904132 0.9917945705699864];

Plots.plot!(plt1, H, l2h1error[:,1], xaxis=:log10, yaxis=:log10, label="L² Error (l=1, p=0)", ls=:dash, lw=2, lc=:blue);
Plots.scatter!(plt1, H, l2h1error[:,1], label="")
Plots.plot!(plt2, H, l2h1error[:,2], xaxis=:log10, yaxis=:log10, label="H¹ Error (l=1, p=0)", ls=:dash, lw=2, lc=:blue);
Plots.scatter!(plt2, H, l2h1error[:,2], label="")

# l = 2, p = 0
l2h1error = [0.10695429648960116 0.22991324795746892;
0.021824703150420847 0.09162428030028616;
0.09747815806612264 0.3077746907927761;
0.4229555559807699 0.6500361500434236;
0.8032011054643603 0.8961955952511019]

Plots.plot!(plt1, H, l2h1error[:,1], xaxis=:log10, yaxis=:log10, label="L² Error (l=2, p=0)", ls=:dash, lw=2, lc=:red);
Plots.scatter!(plt1, H, l2h1error[:,1], label="")
Plots.plot!(plt2, H, l2h1error[:,2], xaxis=:log10, yaxis=:log10, label="H¹ Error (l=2, p=0)", ls=:dash, lw=2, lc=:red);
Plots.scatter!(plt2, H, l2h1error[:,2], label="")

# l = 3, p = 0
l2h1error = [0.10695429648960116 0.22991324795746892;
0.009296041759004649 0.04906522573519141;
0.006585063421319698 0.06449033174671583;
0.043733308726864956 0.2075844168650841;
0.2389855228853218 0.4887254302299412;]

Plots.plot!(plt1, H, l2h1error[:,1], xaxis=:log10, yaxis=:log10, label="L² Error (l=3, p=0)", ls=:dash, lw=2, lc=:olive);
Plots.scatter!(plt1, H, l2h1error[:,1], label="")
Plots.plot!(plt2, H, l2h1error[:,2], xaxis=:log10, yaxis=:log10, label="H¹ Error (l=3, p=0)", ls=:dash, lw=2, lc=:olive);
Plots.scatter!(plt2, H, l2h1error[:,2], label="")

# l = 4, p = 0

l2h1error = [0.10695429648960116 0.22991324795746892;
0.009296041759004649 0.04906522573519141;
0.0016907959489900824 0.018548355576871445;
0.003320265963780901 0.051960655915778656;
0.023140724695286677 0.15156710221864772]

Plots.plot!(plt1, H, l2h1error[:,1], xaxis=:log10, yaxis=:log10, label="L² Error (l=4, p=0)", ls=:dash, lw=2, lc=:orange);
Plots.scatter!(plt1, H, l2h1error[:,1], label="")
Plots.plot!(plt2, H, l2h1error[:,2], xaxis=:log10, yaxis=:log10, label="H¹ Error (l=4, p=0)", ls=:dash, lw=2, lc=:orange);
Plots.scatter!(plt2, H, l2h1error[:,2], label="")


Plots.plot!(plt1, H, H.^3, ls=:dash, lw=1, lc=:black, label="O(h³)")
Plots.plot!(plt2, H, H.^2, ls=:dash, lw=1, lc=:black, label="O(h²)")
Plots.plot(plt1, plt2, layout=(1,2), legend=:bottomright, size=(800,400))