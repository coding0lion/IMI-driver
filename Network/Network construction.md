# Network construction

1、 Construct specific networks

`all_net_demo.m` can generate  **specific  network** includes gene dependency network (GDN), competing endogenous RNA (ceRNA) network, gene co-expression network ( GCN), DNA methylation interaction network (DMIN), co-mutations network (DCMN)., and the detailed construction code for each network can be found in this directory.

2、Construct common networks

- Gene pathway similarity network (GPSN)

  ```
  cd ./Network/pathway_net
  matlab pathway_net.m
  ```

- Protein-protein interaction (PPI) network

  ```
  cd ./Network/ppi
  matlab ppi_net.m
  ```

- Transcriptional regulatory network

  ```
  cd ./Network/tf_target
  matlab tf_net.m
  ```

`demo_pair2table.m` converts neighboring matrices to neighboring tables.

