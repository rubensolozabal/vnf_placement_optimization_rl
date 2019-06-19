include "globals.mzn";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT DATA

int: numServers;
set of int: sServer = 1..numServers;

int: maxSlots;
set of int: sSlot  = 1..maxSlots;
set of int: sSlot_ = 0..maxSlots;

array[sServer] of sSlot: slots;
array[sServer] of int: link_bandwidth;
array[sServer] of int: link_latency;


int: numVnfds;
set of int: sVnfd = 1..numVnfds;
array[sVnfd] of int: vnf_weights;
array[sVnfd] of int: vnf_bandwidth;
array[sVnfd] of int: vnf_latency;

int: chainLen;
set of int: sChain = 1..chainLen;
array[sChain] of sVnfd: chain;


int: p_min;
int: p_slot;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DECISION VARIABLES AND CONSTRAINTS

var int: service_bandwidth;
var int: service_net_latency;
var int: service_cpu_latency;
var int: energy;

array[sChain] of var sServer: placement;
array[sServer] of var sSlot_: occupancy;

array[sServer] of var int: link_used;

array[sChain] of var int: service_bandwidth_;


% Resources

constraint forall (server in sServer) 
(
    occupancy[server] = sum(c in sChain where placement[c]=server) (vnf_weights[chain[c]])
);

constraint forall (server in sServer)
(    
    occupancy[server] <= slots[server]
);



% Bandwidth

constraint forall (c in sChain)
(
	service_bandwidth_[c] = vnf_bandwidth[chain[c]]
);

constraint service_bandwidth = max(service_bandwidth_);

constraint forall (server in sServer)
(
	link_used[server] = sum(c in sChain where placement[c]=server) 

	(
		(
			if c == 1 
				then 1 
			else 
				(
					if placement[c-1] != server then 1 else 0 endif
				)
			
			endif
			+
			if c == chainLen 
				then 1 
			else 
				(
					if placement[c+1] != server then 1 else 0 endif
				)
			
			endif
			
		) * service_bandwidth
	)
);


constraint forall (server in sServer)
(
	link_used[server] <= link_bandwidth[server]
);


% Latency

constraint service_net_latency = sum(c in sChain) 

	(
		(
			if c == 1 
				then 1 
			else 
				(
					if placement[c-1] != placement[c] then 1 else 0 endif
				)
			
			endif
			+
			if c == chainLen 
				then 1 
			else 
				(
					if placement[c+1] != placement[c] then 1 else 0 endif
				)
			
			endif
			
		) * link_latency[placement[c]]
	);


constraint service_cpu_latency = sum(c in sChain) (vnf_latency[chain[c]]);


constraint service_net_latency  <=  service_cpu_latency;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OBJECTIVE FUNCTION

energy = sum(server in sServer) ((if occupancy[server] > 0 then p_min else 0 endif) + occupancy[server]* p_slot);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SOLVE

%solve maximize energy;
solve minimize energy;
%solve satisfy;

output ["\nService = ",show(chain),
        "\nPlacement = ",show(placement),
        "\nEnergy = ",show(energy),
        "\nOccupacy = ", show(occupancy),
        "\nlink_used = ", show(link_used),
        "\nservice_net_latency = ", show(service_net_latency),
        "\nservice_cpu_latency = ", show(service_cpu_latency),
       ];