// ISCAS-85 c17 benchmark circuit
module c17(
    input [4:0] in,
    output [1:0] out
);

    wire n23, n24, n25, n26, n27, n28, n29, n30;

    // Logic gates
    assign n23 = ~(in[0] & in[1]);
    assign n24 = ~(in[3] | in[2]);
    assign n25 = ~(n24 & in[4]);
    assign n26 = ~(n25 | n23);
    assign n27 = ~(n26 & n25);
    assign n28 = ~(n27 | n24);
    assign n29 = ~(n28 & n23);
    assign n30 = ~(n29 | n26);

    assign out[0] = n30;
    assign out[1] = n27;

endmodule
