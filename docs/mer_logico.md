# MER Lógico

  ---------------------------------------------------------------------------------------
  Tabela de Origem  Coluna (FK)     Tabela de Destino     Coluna (PK)     Cardinalidade
  ----------------- --------------- --------------------- --------------- ---------------
  pedidocredito     clienteid       clientes              clienteid       N:1 (Vários
                                                                          pedidos para 1
                                                                          cliente)

  pedidocredito     produtoid       produtosfinanciados   produtoid       N:1 (Vários
                                                                          pedidos para 1
                                                                          produto)

  parcelascredito   solicitacaoid   pedidocredito         solicitacaoid   N:1 (Várias
                                                                          parcelas para 1
                                                                          pedido)
  ---------------------------------------------------------------------------------------
