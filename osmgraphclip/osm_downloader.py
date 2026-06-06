import gzip
import logging
import osmnx as ox
from osmnx._errors import InsufficientResponseError

logger = logging.getLogger(__name__)

class OSMDownloader:

    def __init__(self, lat, lon, dist, output_file, save_road_graph=True):
        self.lat = lat
        self.lon = lon
        self.dist = dist
        self.output_file = output_file
        self.save_road_graph = save_road_graph

    def __call__(self):
        """
        Download OSM data for a given point and distance using osmnx and save files.

        Returns:
            True if at least one artifact (graph or features) was successfully downloaded,
            False otherwise.
        """
        location_point = (self.lat, self.lon)
        logger.info(f"Downloading OSM data for location: {location_point} with distance: {self.dist}m")
        any_downloaded = False

        if self.save_road_graph:
            try:
                graph = ox.graph_from_point(location_point, dist=self.dist, dist_type="bbox", network_type='all')
                graph_filename = f"{self.output_file}.graphml"
                ox.save_graphml(graph, graph_filename)
                logger.info(f"✅ Saved graph to {graph_filename}")
                any_downloaded = True
            except InsufficientResponseError:
                logger.warning(
                    "No graph data returned by OSM for location %s at distance %sm",
                    location_point,
                    self.dist,
                )
            except ValueError as exc:
                if "Found no graph nodes within the requested polygon" in str(exc):
                    logger.warning(
                        "No graph nodes found in requested area for location %s at distance %sm",
                        location_point,
                        self.dist,
                    )
                else:
                    logger.exception("Failed to download graph data for location %s", location_point)
            except Exception:
                logger.exception("Failed to download graph data for location %s", location_point)

        tags = {
            'amenity': True,
            'building': True,
            'highway': True,
            'landuse': True,
            'railway': True,
            'public_transport': True,
            'waterway': True,
            'water': True,
            'telecom': True,
            'leisure': True,
            'natural': True,
            'shop': True,
            'tourism': True,
            'historic': True,
            'man_made': True,
            'power': True,
            'barrier': True,
            'aeroway': True,
            'military': True,
            'office': True,
            'place': True,
            'sport': True,
            'religion': True,
            'boundary': True,
            'surface': True,
            'service': True,
            'denomination': True,
            'craft': True,
            'emergency': True,
            'healthcare': True,
        }
        try:
            features = ox.features_from_point(location_point, dist=self.dist, tags=tags)
            if features is None or features.empty:
                logger.warning("No OSM features returned for location %s", location_point)
                return any_downloaded

            logger.info(f"Downloaded {len(features)} features from OSM.")

            features_filename = f"{self.output_file}_features.geojson.gz"
            with gzip.open(features_filename, "wt", encoding="utf-8") as f:
                f.write(features.to_json(na="drop"))
            logger.info(f"✅ Saved features to {features_filename}")
            any_downloaded = True

            features_with_geometry = features[features.geometry.notna()]
            for geom_type, subset in features_with_geometry.groupby(features_with_geometry.geometry.geom_type):
                if subset.empty:
                    continue
                filename = f"{self.output_file}_{geom_type.lower()}.geojson.gz"
                with gzip.open(filename, "wt", encoding="utf-8") as f:
                    f.write(subset.to_json(na="drop"))
                logger.info(f"✅ Saved {geom_type} features to {filename}")
        except InsufficientResponseError:
            logger.warning(
                "No feature data returned by OSM for location %s at distance %sm",
                location_point,
                self.dist,
            )
        except Exception:
            logger.exception("Failed to download feature data for location %s", location_point)

        return any_downloaded
