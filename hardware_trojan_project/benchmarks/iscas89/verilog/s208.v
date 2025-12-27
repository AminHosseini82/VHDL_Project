// Generic Sequential Circuit
module bench_seq(
    input clk,
    input reset,
    input [11:0] in,
    output [0:0] out
);

    reg [7:0] state, next_state;

    always @(posedge clk or negedge reset) begin
        if (!reset)
            state <= 8'b0;
        else
            state <= next_state;
    end

    always @(*) begin
        next_state = state ^ in[7:0];
    end

    assign out = state[0:0];

endmodule
