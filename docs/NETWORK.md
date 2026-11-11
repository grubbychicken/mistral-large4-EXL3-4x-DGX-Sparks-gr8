# Network setup

The four Sparks need two networks:

1. **Your usual network** (Wi-Fi or Ethernet). `start.sh` uses it for ssh and for downloads.
2. **A ring of QSFP cables** between the Sparks. The engine uses it for the model's data (RDMA, RoCE v2).

## Cables

Each DGX Spark has two QSFP ports. Connect port 0 of each Spark to port 1 of the next Spark, until the ring closes:

```
Spark 0 port 0 ──► Spark 1 port 1
Spark 1 port 0 ──► Spark 2 port 1
Spark 2 port 0 ──► Spark 3 port 1
Spark 3 port 0 ──► Spark 0 port 1
```

It is not important which Spark you call Spark 0. `start.sh` gives the numbers: the Spark where you run it is
Spark 0, and the Spark on its port 0 is Spark 1.

## Addresses

Linux shows each QSFP port as two network devices (for example `enp1s0f0np0` and `enP2p1s0f0np0` for port 0). Give
an IPv4 address to each device that you use. The rules are:

- The two ends of one cable are in the same subnet.
- Each cable has its own subnet (or two subnets, if you give addresses to the two devices of a port).
- The cable subnets are different from the subnet of your usual network.

Example with one `/24` subnet for each device of each cable:

| Cable | Spark and port | Device | Address |
|---|---|---|---|
| 0 → 1 | Spark 0, port 0 | `enp1s0f0np0` / `enP2p1s0f0np0` | `10.20.0.11/24` / `10.20.1.11/24` |
| | Spark 1, port 1 | `enp1s0f1np1` / `enP2p1s0f1np1` | `10.20.0.12/24` / `10.20.1.12/24` |
| 1 → 2 | Spark 1, port 0 | `enp1s0f0np0` / `enP2p1s0f0np0` | `10.20.2.12/24` / `10.20.3.12/24` |
| | Spark 2, port 1 | `enp1s0f1np1` / `enP2p1s0f1np1` | `10.20.2.13/24` / `10.20.3.13/24` |
| 2 → 3 | Spark 2, port 0 | `enp1s0f0np0` / `enP2p1s0f0np0` | `10.20.4.13/24` / `10.20.5.13/24` |
| | Spark 3, port 1 | `enp1s0f1np1` / `enP2p1s0f1np1` | `10.20.4.14/24` / `10.20.5.14/24` |
| 3 → 0 | Spark 3, port 0 | `enp1s0f0np0` / `enP2p1s0f0np0` | `10.20.6.14/24` / `10.20.7.14/24` |
| | Spark 0, port 1 | `enp1s0f1np1` / `enP2p1s0f1np1` | `10.20.6.11/24` / `10.20.7.11/24` |

To set one address so that it stays after a restart (NetworkManager):

```bash
sudo nmcli connection add type ethernet ifname enp1s0f0np0 con-name ring-p0a ipv4.method manual ipv4.addresses 10.20.0.11/24 ipv6.method link-local
```

## What `start.sh` checks

- **Discovery.** Each cable port shows the link-local IPv6 address of the Spark at its other end. `start.sh` asks
  each machine on your usual network that accepts your ssh key for its own addresses, and matches them. The result
  is in `.state/cluster.env`. Delete this file to search again. If your network is larger than 1,024 addresses, or
  discovery fails, set `SPARKS` in `config.env`.
- **Ring.** Each Spark must have cables to exactly two others, and the cables must make one ring.
- **Direction.** Port 0 of each Spark must go to the next Spark.
- **Ports.** Each RDMA port must be `ACTIVE` and must have a RoCE v2 address (`rdma link`, `ip -br address`).
- **Pings.** A full-size ping that cannot be split must pass on each cable address, in the two directions.

## Commands that help

```bash
ip -br address                 # devices, state and addresses
rdma link                      # RDMA ports and their state
ping -M do -s 1472 10.20.0.12  # a full-size ping to the other end of a cable
```
